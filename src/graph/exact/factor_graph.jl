
import Distributions: support, pdf

abstract type FactorGraphNode end 
mutable struct VariableNode <: FactorGraphNode
    variable::Int
    address::Any
    neighbours::Vector{FactorGraphNode} # factors
    support::Vector{Float64}
    function VariableNode(variable::Int, address::Any)
        return new(variable, address, FactorGraphNode[], Float64[])
    end
end
function Base.show(io::IO, variable_node::VariableNode)
    print(io, "VariableNode(", variable_node.variable, ", ", variable_node.address, ")")
end
 
mutable struct FactorNode <: FactorGraphNode
    neighbours::Vector{VariableNode} # variables
    table::Array{Float64}
end
function Base.show(io::IO, factor_node::FactorNode)
    print(io, "FactorNode(", [n.variable for n in factor_node.neighbours], ")")
    # println(io, factor_node.table)
end

function get_support(pgm::PGM, node::VariableNode, parents::Vector{VariableNode})
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    obs = pgm.observed_values[node.variable]
    is_observed = !isnothing(obs)

    supp = Set{Float64}()
    for assignment in Iterators.product([parent.support for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, value) in zip(parents, assignment)
            X[parent.variable] = value
        end
        if is_observed
            v = obs(X)
            push!(supp, v)
        else
            d = dist(X)
            @assert any(isa.(d, [Bernoulli, Binomial, Categorical, DiscreteUniform, Dirac])) d
            push!(supp, collect(support(d))...)
        end
    end
    return sort(collect(supp))
end

# conditional probability distribution p(x|parents(x))
function get_cpd(pgm::PGM, node::VariableNode, parents::Vector{VariableNode})
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]

    cpd = zeros([length(parent.support) for parent in parents]..., length(node.support))
    
    for assignment in Iterators.product([1:length(parent.support) for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, i) in zip(parents, assignment)
            X[parent.variable] = parent.support[i]
        end
        d = dist(X)
        for (i, x) in enumerate(node.support)
            cpd[assignment..., i] = pdf(d, x)
        end
    end

    return cpd
end

function get_factor_graph(pgm::PGM)
    variable_nodes = [VariableNode(i, pgm.addresses[i]) for i in 1:pgm.n_variables]
    factor_nodes = FactorNode[]
    for v in pgm.topological_order
        parents = VariableNode[variable_nodes[x] for (x,y) in pgm.edges if y == v]
        node = variable_nodes[v]
        node.support = get_support(pgm, node, parents)
        if !isnothing(pgm.observed_values[v])
            continue
        end
        cpd = get_cpd(pgm, node, parents)
        factor_node = FactorNode(push!(parents, node), cpd)
        for neighbour in factor_node.neighbours
            push!(neighbour.neighbours, factor_node)
        end
        push!(factor_nodes, factor_node)
    end
    variable_nodes = variable_nodes[isnothing.(pgm.observed_values)]
    return variable_nodes, factor_nodes
end


export get_factor_graph