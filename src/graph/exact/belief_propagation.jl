# In exact inference, the goal is to multiply all factors and sum out all non-relevant variables.
# To speed up this procedure, we want to sum out variables as soon as possible.
# Each node is connected to other nodes via some factors.
# Core idea: for a node x each neighbouring factor s sends a message which is the product of all factors connected
# to x via s and all variables != x summed out.
# On the other hand, each factor constructs this message by multiplying messages from its variables
# and summing out all != x.
# Thus, we can express messages from factors to variables in terms of messages variables to factors and vice-versa.
# To make this procedure work, we have to assume that the factor graph is a tree.
# Thus, we have a root and have a direction in which messages can be sent.
# In the forward pass, all messages are sent towards the root.
# If we also care about the marginals of other nodes, we sent messages in the reverse direction.

mutable struct BeliefNode
    node::FactorGraphNode
    parent::Union{Nothing,BeliefNode}
    parent_index::Int
    neighbours::Vector{BeliefNode}
    message_sent::Bool
    messages::Vector{Vector{Float64}}

    # Constructor already creates tree from factor graph.
    # Make sure that factor graph is indeed tree.
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


# Method to check whether a factor graph is a tree.
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

# Method to check whether a PGM is a tree.
function is_tree(pgm::PGM)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    return is_tree(variable_nodes, factor_nodes)
end

export is_tree

# Returns all variable nodes from the belief tree.
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

function belief_propagation(pgm::PGM; all_marginals::Bool=false)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    @assert is_tree(variable_nodes, factor_nodes)
    belief_propagation(return_factor, all_marginals)
end

function get_blief_tree(return_factor::FactorNode)
    # we set the return factor as root, as we mainly care about its marginals.
    root = BeliefNode(return_factor, nothing)
    return root
end
export get_blief_tree

function belief_propagation(return_factor::FactorNode, all_marginals::Bool)
    root = BeliefNode(return_factor, nothing)
    return belief_propagation(root, return_factor, all_marginals)
end

function belief_propagation(root::BeliefNode, return_factor::FactorNode, all_marginals::Bool; with_division::Bool=false)
    # print_belief_tree(root)
    
    # Run forward pass. As the return factor has no parents to send a message to,
    # its message actually corresponds to the joint density with all variables summed out - i.e. the evidence. 
    res = forward(root)
    evidence = exp(res[1])

    
    # [root.neighbours].node are root.node.neighbours
    # p(X_f) = f(X_f) ∏_{x ∈ ne(f)} μ_{x → f}(x)
    # for return factor f(X) = 1
    return_table = zeros(Tuple([length(message) for message in root.messages]))
    shape = ones(Int, length(root.neighbours))
    for (i, message) in enumerate(root.messages)
        # (1, ...,         i,       i+1, ... )
        # (1, ..., length(message), 1, ..., 1)
        shape[i] = length(message)
        return_table .+= reshape(message, Tuple(shape)) # broadcasting -> factor product
        shape[i] = 1
    end
    # true return factor is used in backward pass, we should only modify them now
    _return_factor = FactorNode(return_factor.neighbours, return_table)

    if all_marginals
        # only need backward pass if we want to evaluate all marginals
        if with_division
            backward_with_division(root)
        else
            backward(root)
        end

        variable_nodes = get_variable_nodes(root)
        marginals = Vector{Tuple{Int, Any, Vector{Float64}}}(undef, length(variable_nodes))
        # p(x) = ∑_{X - x} p(X) = ∏_{s ∈ ne(x)} μ_{f_s → x}(x)
        for (i,v) in enumerate(variable_nodes)
            varnode = v.node
            table = exp.(sum(v.messages))
            table /= sum(table)
            marginals[i] = (varnode.variable, varnode.address, table)
        end

        return _return_factor, evidence, marginals
    else
        return _return_factor, evidence
    end
end

# message from children to parents, start at leaves
function forward(belief_node::BeliefNode)
    @assert !belief_node.message_sent
    belief_node.message_sent = true

    # Edge case: belief node with only one variable, has to be parent node
    if length(belief_node.neighbours) == 1 && !isnothing(belief_node.parent)
        if belief_node.node isa VariableNode
            println("Who am I? ", belief_node) # cant really happen?
            message = zeros(length(belief_node.node.support))
        else
            # factor node with one variable == parent
            message = belief_node.node.table
            # parent is VariableNode
            @assert length(message) == length(belief_node.parent.node.support)
        end
        return message
    end

    if belief_node.node isa VariableNode
        # message to FactorNode μ_{x → f}(x)
        # x = belief_node
        # f = belief_node.parent
        # product of all children messages
        # ∏_{i ∈ ne(x) - f} μ_{f_i → x}(x)
        # μ_{f_i → x} = belief_node.messages[i]
        # have all the same support belief_node.node.support x
        message = zeros(length(belief_node.node.support))
        for (i, child) in enumerate(belief_node.neighbours)
            # child is FactorNode
            child == belief_node.parent && continue # ne(x) - f
            child_message = forward(child)
            belief_node.messages[i] = child_message
            message .+= child_message # no broadcasting
        end
    else
        # message to VariableNode μ_{f → x}(x)
        # f = belief_node
        # x = belief_node.parent
        # first compute the product of all messages from children
        # ∏_{i ∈ ne(f) - x} μ_{x_i → f}(x_i)
        # has support x_1 × x_2 × ... = scope(f) - x
        message_table = zeros(Tuple([length(child.node.support) for child in belief_node.neighbours if child != belief_node.parent]))
        # factor product is computed by broadcasting + in log space
        shape = ones(Int, ndims(message_table))
        j = 1
        for (i, child) in enumerate(belief_node.neighbours)
            # child is VariableNode
            child == belief_node.parent && continue # ne(f) - x
            child_message = forward(child)
            belief_node.messages[i] = child_message
            @assert length(child_message) == length(child.node.support)
            # each variables has its own dimension j
            # we reshape the one dimensional message to
            # (1,...,1,length(child_message),1,...,1)
            #              j-th index
            shape[j] = length(child_message)
            message_table .+= reshape(child_message, Tuple(shape)) # broadcasting -> factor product
            shape[j] = 1
            j += 1
        end
        # we put result in FactorNode
        # child_variables = scope(belief_node.node) - x 
        child_variables = VariableNode[child.node for child in belief_node.neighbours if child != belief_node.parent]
        # as all factor nodes are sorted and we do not change the order the message_factor is also sorted
        @assert issorted(child_variables)

        message_factor = FactorNode(child_variables, message_table)

        # next we multiple the factor f to the intermediate result
        # scope(message_factor) ∪ {x} = child_variables ∪ {x} =  scope(belief_node.node)
        message_factor = factor_product(belief_node.node, message_factor)
        # resulting message_factor has same dimension /scope as belief_node
        @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table)) (message_factor, belief_node.node)
        @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)

        # now we sum out all variables except for x
        message_factor = factor_sum(message_factor, child_variables)
        if !isnothing(belief_node.parent)
            # the resulting message_factor is one dimensional
            @assert length(size(message_factor.table)) == 1
            # has the same support as parent x
            @assert length(message_factor.table) == length(belief_node.parent.node.support)
        end

        # the resulting message is ∑_{ne(f)-x} ( f(X) ∏_{i ∈ ne(f) - x} μ_{x_i → f}(x_i) )
        message = message_factor.table
    end

    return message
end

function backward(belief_node::BeliefNode)
    @assert belief_node.message_sent
    # belief_node has received all messages from children !and! parent

    # we do the same computations as in forward, but now for every child instead of just the parent
    if belief_node.node isa VariableNode
        for (i, child) in enumerate(belief_node.neighbours)
            child == belief_node.parent && continue # we do not send a message to parent
            # message to FactorNode child i 
            message = zeros(length(belief_node.node.support))
            for child_message in belief_node.messages
                # child is FactorNode
                message .+= child_message # no broadcasting
            end
            # now instead of summing every message up except from parent,
            # we some everything up except from the child we send the message to
            message .-= belief_node.messages[i]

            # put the message into child
            child.messages[child.parent_index] = message
            backward(child)
        end
    else
        for child in belief_node.neighbours
            child == belief_node.parent && continue # we do not send a message to parent
            # message to VariableNode child
            message_vars = VariableNode[neighbour.node for neighbour in belief_node.neighbours if neighbour != child]
            # as all factor nodes are sorted and we do not change the order the message_factor is also sorted
            @assert issorted(message_vars)

            message_table = zeros(Tuple([length(variable.support) for variable in message_vars]))
            shape = ones(Int, ndims(message_table))
            j = 1
            # same spiel as in forward pass
            for (i, neighbour) in enumerate(belief_node.neighbours)
                # child is VariableNode
                neighbour == child && continue # we do not multiply message of child we now send a message to
                child_message = belief_node.messages[i] # = Factor([child], belief_node.messages[i])
                shape[j] = length(child_message)
                message_table .+= reshape(child_message, Tuple(shape)) # broadcasting -> factor product
                shape[j] = 1
                j += 1
            end

            message_factor = FactorNode(message_vars, message_table)

            message_factor = factor_product(belief_node.node, message_factor)
            @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table)) (message_factor, belief_node.node)
            @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)

            message_factor = factor_sum(message_factor, message_vars)
            @assert length(size(message_factor.table)) == 1
            @assert length(message_factor.table) == length(child.node.support)

            message = message_factor.table

            # put the message into child
            child.messages[child.parent_index] = message
            backward(child)
        end
    end
end

function backward_with_division(belief_node::BeliefNode)
    @assert belief_node.message_sent
    # belief_node has received all messages from children !and! parent

    # we do the same computations as in forward, but now for every child instead of just the parent
    if belief_node.node isa VariableNode
        base_message = sum(belief_node.messages)
        for (i, child) in enumerate(belief_node.neighbours)
            child == belief_node.parent && continue # we do not send a message to parent
            # message to FactorNode child i 
            message = copy(base_message)
            # now instead of summing every message up except from parent,
            # we some everything up except from the child we send the message to
            message .-= belief_node.messages[i]

            # put the message into child
            child.messages[child.parent_index] = message
            backward_with_division(child)
        end
    else

        base_message_table = zeros(Tuple([length(neighbour.node.support) for neighbour in belief_node.neighbours]))
        shape = ones(Int, ndims(base_message_table))
        # same spiel as in forward pass
        for (j, neighbour) in enumerate(belief_node.neighbours)
            child_message = belief_node.messages[j]
            shape[j] = length(child_message)
            base_message_table .+= reshape(child_message, Tuple(shape)) # (1) broadcasting -> factor product
            shape[j] = 1
        end

        selection::Vector{Any} = fill(Colon(), ndims(base_message_table))
        for (i, child) in enumerate(belief_node.neighbours)
            child == belief_node.parent && continue # we do not send a message to parent
            # message to VariableNode child
            message_vars = VariableNode[neighbour.node for neighbour in belief_node.neighbours if neighbour != child]
            
            selection[i] = 1
            for j in eachindex(belief_node.messages[i])
                if belief_node.messages[i][j] != -Inf
                    selection[i] = j
                    break
                end
            end
            # belief_node.messages[i][selection[i]] isa Float64
            # we select one row in dimension i, `selection[i]`, and substruct the value of belief_node.messages[i][selection[i]]
            # this undos the addition in (1)
            # if all(belief_node.messages[i] .== -Inf) then all(base_message_table .== -Inf)
            message_factor = FactorNode(message_vars,
                broadcast(factor_div_op, base_message_table[selection...], belief_node.messages[i][selection[i]])
            )

            message_factor = factor_product(belief_node.node, message_factor)
            @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table)) (message_factor, belief_node.node)
            @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)

            message_factor = factor_sum(message_factor, message_vars)
            @assert length(size(message_factor.table)) == 1
            @assert length(message_factor.table) == length(child.node.support)

            message = message_factor.table

            # put the message into child
            child.messages[child.parent_index] = message
            selection[i] = Colon()

            backward_with_division(child)
        end
    end
end

export BeliefNode, print_belief_tree, add_return_factor!, belief_propagation