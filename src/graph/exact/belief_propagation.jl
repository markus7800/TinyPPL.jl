

struct Message
    from::FactorGraphNode
    to::FactorGraphNode
    table::Array{Float64}
end

mutable struct BeliefNode
    node::FactorGraphNode
    parent::Union{Nothing,BeliefNode}
    children::Vector{BeliefNode}
    message_sent::Bool
    belief::Array{Float64}
    function BeliefNode(node::FactorGraphNode, parent::Union{Nothing,FactorGraphNode})
        children = [BeliefNode(n, node) for n in node.neighbours if n != parent]
        @assert length(children) + !isnothing(parent) == length(node.neighbours)
        message_sent = false
        if node isa VariableNode
            belief = zeros(length(node.support))
        else
            belief = zeros(size(node.table))
        end
        # parent variable is set in parent constructor
        this = new(node, nothing, children, message_sent, belief)
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

function belief_propagation(pgm::PGM)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    root = BeliefNode(return_factor, nothing)
    print_belief_tree(root)
    evidence = exp(forward(root)[1])
    return_factor.table .= root.belief
    return return_factor, evidence
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
            message .+= forward(child)
        end
        belief_node.belief .+= message
    else
        # message to VariableNode
        message_table = zeros([length(child.node.support) for child in belief_node.children]...)
        shape = ones(Int, length(belief_node.children))
        for (i, child) in enumerate(belief_node.children)
            # child is VariableNode
            child_message = forward(child)
            @assert length(child_message) == length(child.node.support)
            shape[i] = length(child_message)
            message_table .+= reshape(child_message, shape...) # broadcasting
            shape[i] = 1
        end
        child_variables = [child.node for child in belief_node.children]
        message_factor = FactorNode(child_variables, message_table)
        println("belief node: ", belief_node.node)
        println("message_factor 1: ", message_factor)
        message_factor = factor_product(belief_node.node, message_factor) # left argument dictates order of common nodes
        println("message_factor 2: ", message_factor)
        @assert prod(size(message_factor.table)) == prod(size(belief_node.node.table))
        @assert length(message_factor.neighbours ∩ belief_node.node.neighbours) == length(belief_node.node.neighbours)
        println("belief: ", size(belief_node.belief))
        belief_node.belief .+= factor_permute_vars(message_factor, belief_node.node.neighbours).table
        message_factor = factor_sum(message_factor, child_variables)
        if !isnothing(belief_node.parent)
            @assert length(size(message_factor.table)) == 1 # should factor out all variables except parent
            @assert length(message_factor.table) == length(belief_node.parent.node.support)
        end
        message = message_factor.table
    end

    return message
end

# function backward(node::BeliefNode)
# simply add backward messages to belief?
# end

export BeliefNode, print_belief_tree, add_return_factor!, belief_propagation