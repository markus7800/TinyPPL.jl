
mutable struct BeliefNode
    node::FactorGraphNode
    parent::Union{Nothing,BeliefNode}
    children::Vector{BeliefNode}
    message_sent::Bool
    messages::Vector{Vector{Float64}}
    function BeliefNode(node::FactorGraphNode, parent::Union{Nothing,FactorGraphNode})
        children = [BeliefNode(n, node) for n in node.neighbours if n != parent]
        @assert length(children) + !isnothing(parent) == length(node.neighbours)
        message_sent = false
        if node isa VariableNode
            belief = zeros(length(node.support))
        else
            belief = zeros(size(node.table))
        end
        messages = Vector{Vector{Float64}}(undef, length(children) + !isnothing(parent))
        # parent variable is set in child constructor
        this = new(node, nothing, children, message_sent, messages)
        for child in children
            child.parent = this
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

function add_return_factor!(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    variable_to_node = Dict(node.variable=>node for node in variable_nodes)
    return_variables = [variable_to_node[v] for v in return_expr_variables(pgm)]
    return_factor = FactorNode(return_variables, zeros([length(node.support) for node in return_variables]...))
    for variable in return_variables
        push!(variable.neighbours, return_factor)
    end
    push!(factor_nodes, return_factor)
    return return_factor
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
    for child in belief_node.children
        get_variable_nodes(child, variable_nodes)
    end
    return variable_nodes
end

function belief_propagation(pgm::PGM, all_marginals=false)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    @assert is_tree(variable_nodes, factor_nodes)

    root = BeliefNode(return_factor, nothing)
    # print_belief_tree(root)
    res = forward(root)
    evidence = exp(res[1])

    
    # [root.children].node are root.node.neighbours
    table = zeros([length(message) for message in root.messages]...)
    shape = ones(Int, length(root.children))
    for (i, message) in enumerate(root.messages)
        shape[i] = length(message)
        table .+= reshape(message, shape...) # broadcasting -> factor product
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

        return_factor.table .= table

        return return_factor, evidence, marginals
    else

        return_factor.table .= table

        return return_factor, evidence
    end
end

# message from children to parents, start at leaves
function forward(belief_node::BeliefNode)
    @assert !belief_node.message_sent
    belief_node.message_sent = true

    if isempty(belief_node.children) # has to have parent
        if belief_node.node isa VariableNode
            message = zeros(size(belief_node.parent.table))
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
        for (i, child) in enumerate(belief_node.children)
            # child is FactorNode
            child_message = forward(child)
            belief_node.messages[i] = child_message
            message .+= child_message # no broadcasting
        end
    else
        # message to VariableNode
        message_table = zeros([length(child.node.support) for child in belief_node.children]...)
        shape = ones(Int, length(belief_node.children))
        for (i, child) in enumerate(belief_node.children)
            # child is VariableNode
            child_message = forward(child)
            belief_node.messages[i] = child_message
            @assert length(child_message) == length(child.node.support)
            shape[i] = length(child_message)
            message_table .+= reshape(child_message, shape...) # broadcasting -> factor product
            shape[i] = 1
        end
        child_variables = [child.node for child in belief_node.children]
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
        for (i, child) in enumerate(belief_node.children)
            # message to FactorNode child i 
            message = zeros(length(belief_node.node.support))
            for child_message in belief_node.messages
                # child is FactorNode
                message .+= child_message # no broadcasting
            end
            message .-= belief_node.messages[i]

            child.messages[end] = message # reserved for parent / should be undef
            backward(child)
        end
    else
        neighbours = BeliefNode[child for child in belief_node.children]
        if !isnothing(belief_node.parent)
            push!(neighbours, belief_node.parent)
        end
        mask = trues(length(neighbours))
        @assert length(mask) == length(belief_node.messages)

        for (i, child) in enumerate(belief_node.children)
            # message to VariableNode child i
            mask[i] = false
            message_vars = [neighbour.node for neighbour in neighbours[mask]]
            message_table = zeros([length(variable.support) for variable in message_vars]...)
            shape = ones(Int, length(size(message_table)))
            for (j, child_message) in enumerate(belief_node.messages[mask])
                # child is VariableNode
                shape[j] = length(child_message)
                message_table .+= reshape(child_message, shape...) # broadcasting -> factor product
                shape[j] = 1
            end
            mask[i] = true

            message_factor = FactorNode(message_vars, message_table)

            message_factor = factor_product(belief_node.node, message_factor)
            @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table))
            @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)

            message_factor = factor_sum(message_factor, message_vars)
            @assert length(size(message_factor.table)) == 1 # should sum out all variables except child
            @assert length(message_factor.table) == length(child.node.support)

            message = message_factor.table

            child.messages[end] = message # reserved for parent / should be undef
            backward(child)
        end
    end
end

export BeliefNode, print_belief_tree, add_return_factor!, belief_propagation