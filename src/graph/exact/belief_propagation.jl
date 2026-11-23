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

# All traversals of the belief tree are iterative (see `preorder`), since recursion overflows the stack
# for deep trees (e.g. chains with ~10^5 variables).
mutable struct BeliefNode
    node::FactorGraphNode
    parent::Union{Nothing,BeliefNode}
    parent_index::Int # index of parent in neighbours
    index_in_parent::Int # index of this node in parent.neighbours
    neighbours::Vector{BeliefNode}
    message_sent::Bool
    messages::Vector{Vector{Float64}} # messages[i] is the message from neighbours[i]

    # creates node without neighbours (they are set by the constructor below)
    function BeliefNode(node::FactorGraphNode, parent::Union{Nothing,BeliefNode}, index_in_parent::Int)
        this = new()
        this.node = node
        this.parent = parent
        this.parent_index = 0
        this.index_in_parent = index_in_parent
        this.neighbours = Vector{BeliefNode}(undef, length(node.neighbours))
        this.message_sent = false
        this.messages = Vector{Vector{Float64}}(undef, length(node.neighbours))
        return this
    end
end

# Constructor already creates tree from factor graph.
# Make sure that factor graph is indeed tree.
function BeliefNode(node::FactorGraphNode, parent::Union{Nothing,BeliefNode})
    root = BeliefNode(node, parent, 0)
    stack = BeliefNode[root]
    while !isempty(stack)
        this = pop!(stack)
        for (i, n) in enumerate(this.node.neighbours)
            if !isnothing(this.parent) && this.parent.node == n
                this.parent_index = i
                this.neighbours[i] = this.parent
            else
                child = BeliefNode(n, this, i)
                this.neighbours[i] = child
                push!(stack, child)
            end
        end
    end
    return root
end

# Nodes of the belief tree in depth-first pre-order (parent before children, children in order of neighbours),
# the same order as was visited by recursion.
function preorder(root::BeliefNode)
    nodes = BeliefNode[]
    stack = BeliefNode[root]
    while !isempty(stack)
        node = pop!(stack)
        push!(nodes, node)
        for child in Iterators.reverse(node.neighbours)
            child === node.parent && continue
            push!(stack, child)
        end
    end
    return nodes
end

function Base.show(io::IO, belief_node::BeliefNode)
    print(io, "BeliefNode(", belief_node.node, ")")
end

function print_belief_tree(root::BeliefNode)
    depth = Dict{BeliefNode,Int}(root => 0)
    for node in preorder(root)
        if !isnothing(node.parent) && node !== root
            depth[node] = depth[node.parent] + 1
        end
        println("  "^depth[node], node.node)
    end
end


# Method to check whether a factor graph is a tree.
function is_tree(variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    # depth-first search from start, a cycle exists if we reach a visited node that is not the parent
    # works only for undirected graphs (Because DFS dives as deep as possible along one branch before
    # backtracking, any previously visited node you encounter in an undirected graph must be an ancestor
    # in the current recursion stack, which guarantees a closed path back to it)
    function is_cycle(start, visited)
        visited[start] = true
        # stack is (node, parent of node), parent is also neighbour of node
        stack = Tuple{FactorGraphNode,Union{Nothing,FactorGraphNode}}[(start, nothing)]
        while !isempty(stack)
            node, parent = pop!(stack)
            for neighbour in node.neighbours
                if visited[neighbour] == false
                    visited[neighbour] = true
                    push!(stack, (neighbour, node))
                elseif neighbour != parent
                    return true
                end
            end
        end
        return false
    end
    
    visited = Dict(v => false for v in vcat(variable_nodes, factor_nodes))
    if is_cycle(factor_nodes[end], visited)
        @debug "factor graph has a cycle" start=factor_nodes[end]
        return false
    end

    for (n, b) in visited
        if !b
            @debug "factor graph is not connected" start=factor_nodes[end] not_connected=n
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
function get_variable_nodes(root::BeliefNode)
    return BeliefNode[belief_node for belief_node in preorder(root) if belief_node.node isa VariableNode]
end

function belief_propagation(pgm::PGM; calibrate_tree::Bool=false, return_factor_as_root::Bool=false)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    if return_factor_as_root
        # we set the return factor as root, as we mainly care about its marginals.
        return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
        root_factor = return_factor
    else
        root_factor = factor_nodes[1]
    end
    @assert is_tree(variable_nodes, factor_nodes)
    belief_propagation(root_factor, calibrate_tree)
end

function get_belief_tree(root_factor::FactorNode)
    root = BeliefNode(root_factor, nothing)
    return root
end
export get_belief_tree

function belief_propagation(root_factor::FactorNode, calibrate_tree::Bool)
    root = BeliefNode(root_factor, nothing)
    return belief_propagation(root, root_factor, calibrate_tree)
end

struct BeliefPropagationResult
    is_calibrated::Bool
    root::BeliefNode
    evidence::Float64 # may underflow, use log_evidence
    log_evidence::Float64
end
Base.show(io::IO, tree::BeliefPropagationResult) = print(io, "BeliefPropagationResult()")

function belief_propagation(root::BeliefNode, root_factor::FactorNode, calibrate_tree::Bool; with_division::Bool=false)
    
    # Run forward pass. As the root factor has no parents to send a message to,
    # its message actually corresponds to the joint density with all variables summed out - i.e. the evidence. 
    res = forward(root)
    log_evidence = res[1]

    if calibrate_tree
        # only need backward pass if we want to evaluate all marginals
        if with_division
            backward_with_division(root)
        else
            backward(root)
        end
    end
    return BeliefPropagationResult(calibrate_tree, root, exp(log_evidence), log_evidence)
end

function _compute_factor_from_messages(beliefnode::BeliefNode)
    @assert beliefnode.node isa FactorNode
    # p(X_f) = f(X_f) ∏_{x ∈ ne(f)} μ_{x → f}(x)
    # root_table = zeros(Tuple([length(message) for message in root.messages]))
    table = copy(beliefnode.node.table)
    shape = ones(Int, length(beliefnode.neighbours))
    for (i, message) in enumerate(beliefnode.messages)
        # (1, ...,         i,       i+1, ... )
        # (1, ..., length(message), 1, ..., 1)
        shape[i] = length(message)
        table .+= reshape(message, Tuple(shape)) # broadcasting -> factor product
        shape[i] = 1
    end
    return FactorNode(beliefnode.node.neighbours, table)
end

# posterior over the scope of the root factor, see get_posterior(::VariableEliminationResult)
function get_posterior(res::BeliefPropagationResult)
    # [root.neighbours].node are root.node.neighbours
    root_factor = _compute_factor_from_messages(res.root)
    # sum(exp, root_factor.table) should be equal to evidence
    return exp.(root_factor.table .- res.log_evidence)
end

function get_marginals(res::BeliefPropagationResult)
    @assert res.is_calibrated
    variable_nodes = get_variable_nodes(res.root)
    marginals = Vector{Tuple{VariableNode, Vector{Float64}}}(undef, length(variable_nodes))
    # p(x) = ∑_{X - x} p(X) = ∏_{s ∈ ne(x)} μ_{f_s → x}(x)
    for (i,v) in enumerate(variable_nodes)
        varnode = v.node
        log_table = sum(v.messages)
        table = exp.(log_table .- logsumexp(log_table))
        marginals[i] = (varnode, table)
    end
    return marginals
end
export get_marginals

# message from children to parents, start at leaves
# (nodes in reverse pre-order: all children are processed before their parent)
# returns the message of the root, i.e. the log evidence
function forward(root::BeliefNode)
    message = Float64[]
    for belief_node in Iterators.reverse(preorder(root))
        message = forward_message(belief_node)
        if belief_node !== root
            belief_node.parent.messages[belief_node.index_in_parent] = message
        end
    end
    return message
end

# computes the message to the parent from the messages of the children (stored in belief_node.messages)
function forward_message(belief_node::BeliefNode)
    @assert !belief_node.message_sent
    belief_node.message_sent = true

    # Edge case: belief node with only one variable, has to be parent node
    if length(belief_node.neighbours) == 1 && !isnothing(belief_node.parent)
        if belief_node.node isa VariableNode
            # variable whose only factor is the parent (e.g. a latent variable without children),
            # sends the uniform message (log 1 = 0)
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
            child_message = belief_node.messages[i] # set by forward pass of child
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
            child_message = belief_node.messages[i] # set by forward pass of child
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

# messages from parents to children, start at root (pre-order: parent before children)
function backward(root::BeliefNode)
    for belief_node in preorder(root)
        backward_messages(belief_node)
    end
end

# sends messages to all children (stored in child.messages)
function backward_messages(belief_node::BeliefNode)
    @assert belief_node.message_sent
    # belief_node has received all messages from children !and! parent

    # we do the same computations as in forward, but now for every child instead of just the parent
    if belief_node.node isa VariableNode
        for (i, child) in enumerate(belief_node.neighbours)
            child == belief_node.parent && continue # we do not send a message to parent
            # message to FactorNode child i 
            message = zeros(length(belief_node.node.support))
            for (j, child_message) in enumerate(belief_node.messages)
                # child is FactorNode
                i == j && continue
                # now instead of summing every message up except from parent,
                # we some everything up except from the child we send the message to
                message .+= child_message # no broadcasting
            end
            # message .-= belief_node.messages[i]

            # put the message into child
            child.messages[child.parent_index] = message
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
        end
    end
end

function backward_with_division(root::BeliefNode)
    for belief_node in preorder(root)
        backward_messages_with_division(belief_node)
    end
end

function backward_messages_with_division(belief_node::BeliefNode)
    @assert belief_node.message_sent
    # belief_node has received all messages from children !and! parent

    # we do the same computations as in forward, but now for every child instead of just the parent
    if belief_node.node isa VariableNode
        base_message = sum(belief_node.messages)
        for (i, child) in enumerate(belief_node.neighbours)
            child == belief_node.parent && continue # we do not send a message to parent
            # message to FactorNode child i 
            # now instead of summing every message up except from parent,
            # we some everything up except from the child we send the message to
            # factor_div_op uses the convention 0 / 0 = 0 (-Inf - -Inf = -Inf), PGM 10.3.1
            message = broadcast(factor_div_op, base_message, belief_node.messages[i])

            # put the message into child
            child.messages[child.parent_index] = message
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
        end
    end
end

export BeliefNode, print_belief_tree, add_return_factor!, belief_propagation