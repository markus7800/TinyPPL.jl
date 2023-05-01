
mutable struct BeliefNode
    node::FactorGraphNode
    parent::Union{Nothing,BeliefNode}
    parent_index::Int
    neighbours::Vector{BeliefNode}
    message_sent::Bool
    messages::Vector{Vector{Float64}}
    function BeliefNode(node::FactorGraphNode, parent::Union{Nothing,BeliefNode})
        this = new()
        this.node = node
        this.parent = parent
        this.parent_index = 0
        this.neighbours = Vector{Vector{BeliefNode}}(undef, length(node.neighbours))
        this.message_sent = false
        this.messages = Vector{Vector{Float64}}(undef, length(node.neighbours))

        for (i, n) in enumerate(node.neighbours)
            if !isnothing(parent) && parent.node == n
                this.parent_index = i
                this.neighbours[i] = parent
            else
                this.neighbours[i] = BeliefNode(n, this)
            end
        end

        return this
    end
end

function Base.show(io::IO, belief_node::BeliefNode)
    print(io, "BeliefNode(", belief_node.node, ")")
end

function print_belief_tree(root::BeliefNode, tab="")
    println(tab, root.node)
    for child in root.children
        print_belief_tree(child, tab*"  ")
    end
end

function is_tree(variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})

    function is_cycle(node, visited, parent, tab="")
        visited[node] = true

        for neighbour in node.neighbours
            if visited[neighbour] == false
                if is_cycle(neighbour, visited, node, tab*"  ")
                    return true
                end
            elseif neighbour != parent
                return true
            end
        end

        return false
    end
    
    visited = Dict(v => false for v in vcat(variable_nodes, factor_nodes))
    if is_cycle(factor_nodes[end], visited, nothing)
        return false
    end

    for (_, b) in visited
        if !b
            # not connected
            return false
        end
    end

    return true
end

function is_tree(pgm::PGM)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    return is_tree(variable_nodes, factor_nodes)
end

export is_tree

function get_variable_nodes(belief_node::BeliefNode, variable_nodes=BeliefNode[])
    if belief_node.node isa VariableNode
        push!(variable_nodes, belief_node)
    end
    for child in belief_node.neighbours
        child == belief_node.parent && continue
        get_variable_nodes(child, variable_nodes)
    end
    return variable_nodes
end
function belief_propagation(pgm::PGM; all_marginals=false)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    @assert is_tree(variable_nodes, factor_nodes)
    belief_propagation(return_factor, all_marginals)
end

function belief_propagation(return_factor::FactorNode, all_marginals::Bool)
    root = BeliefNode(return_factor, nothing)
    # print_belief_tree(root)
    res = forward(root)
    evidence = exp(res[1])

    
    # [root.neighbours].node are root.node.neighbours
    return_table = zeros([length(message) for message in root.messages]...)
    shape = ones(Int, length(root.neighbours))
    for (i, message) in enumerate(root.messages)
        shape[i] = length(message)
        return_table .+= reshape(message, shape...) # broadcasting -> factor product
        shape[i] = 1
    end

    if all_marginals
        backward(root)
        variable_nodes = get_variable_nodes(root)
        marginals = Vector{Tuple{Int, Any, Vector{Float64}}}(undef, length(variable_nodes))
        for (i,v) in enumerate(variable_nodes)
            varnode = v.node
            table = exp.(sum(v.messages))
            table /= sum(table)
            marginals[i] = (varnode.variable, varnode.address, table)
        end

        return_factor.table .= return_table

        return return_factor, evidence, marginals
    else

        return_factor.table .= return_table

        return return_factor, evidence
    end
end

# message from children to parents, start at leaves
function forward(belief_node::BeliefNode)
    @assert !belief_node.message_sent
    belief_node.message_sent = true

    if length(belief_node.neighbours) == 1 && !isnothing(belief_node.parent) # has to be parent
        if belief_node.node isa VariableNode
            message = zeros(length(belief_node.node.support))
        else
            message = belief_node.node.table
            # parent is VariableNode
            @assert length(message) == length(belief_node.parent.node.support)
        end
        return message
    end

    if belief_node.node isa VariableNode
        # message to FactorNode
        message = zeros(length(belief_node.node.support))
        for (i, child) in enumerate(belief_node.neighbours)
            # child is FactorNode
            child == belief_node.parent && continue
            child_message = forward(child)
            belief_node.messages[i] = child_message
            message .+= child_message # no broadcasting
        end
    else
        # message to VariableNode
        message_table = zeros([length(child.node.support) for child in belief_node.neighbours if child != belief_node.parent]...)
        shape = ones(Int, ndims(message_table))
        j = 1
        for (i, child) in enumerate(belief_node.neighbours)
            # child is VariableNode
            child == belief_node.parent && continue
            child_message = forward(child)
            belief_node.messages[i] = child_message
            @assert length(child_message) == length(child.node.support)
            shape[j] = length(child_message)
            message_table .+= reshape(child_message, shape...) # broadcasting -> factor product
            shape[j] = 1
            j += 1
        end
        child_variables = [child.node for child in belief_node.neighbours if child != belief_node.parent]
        message_factor = FactorNode(child_variables, message_table)

        message_factor = factor_product(belief_node.node, message_factor)
        @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table))
        @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)

        message_factor = factor_sum(message_factor, child_variables)
        if !isnothing(belief_node.parent)
            @assert length(size(message_factor.table)) == 1 # should sum out all variables except parent
            @assert length(message_factor.table) == length(belief_node.parent.node.support)
        end

        message = message_factor.table
    end

    return message
end

function backward(belief_node::BeliefNode)
    @assert belief_node.message_sent
    # belief_node has received all messages from children !and! parent

    if belief_node.node isa VariableNode
        for (i, child) in enumerate(belief_node.neighbours)
            child == belief_node.parent && continue
            # message to FactorNode child i 
            message = zeros(length(belief_node.node.support))
            for child_message in belief_node.messages
                # child is FactorNode
                message .+= child_message # no broadcasting
            end
            message .-= belief_node.messages[i]

            child.messages[child.parent_index] = message
            backward(child)
        end
    else
        mask = trues(length(belief_node.neighbours))

        for child in belief_node.neighbours
            child == belief_node.parent && continue
            # message to VariableNode child
            message_vars = VariableNode[neighbour.node for neighbour in belief_node.neighbours if neighbour != child]
            message_table = zeros([length(variable.support) for variable in message_vars]...)
            shape = ones(Int, ndims(message_table))
            j = 1
            for (i, neighbour) in enumerate(belief_node.neighbours)
                # child is VariableNode
                neighbour == child && continue
                child_message = belief_node.messages[i]
                shape[j] = length(child_message)
                message_table .+= reshape(child_message, shape...) # broadcasting -> factor product
                shape[j] = 1
                j += 1
            end

            message_factor = FactorNode(message_vars, message_table)
            # println("neighbours: ", neighbours)
            # println("message_factor_1: ", message_factor)

            message_factor = factor_product(belief_node.node, message_factor)
            # println("message_factor_2: ", message_factor)
            @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table)) (message_factor, belief_node.node)
            @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)

            message_factor = factor_sum(message_factor, message_vars)
            @assert length(size(message_factor.table)) == 1 # should sum out all variables except child
            @assert length(message_factor.table) == length(child.node.support)

            message = message_factor.table

            child.messages[child.parent_index] = message
            backward(child)
        end
    end
end

export BeliefNode, print_belief_tree, add_return_factor!, belief_propagation