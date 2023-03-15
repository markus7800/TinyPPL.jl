
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
    print(io, "FactorNode(", [n.address for n in factor_node.neighbours], ")")
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

# conditional probability distribution p(x|parents(x)) for unobserved
# likelihood p(y=e|parents(y)) for observed
function get_table(pgm::PGM, node::VariableNode, parents::Vector{VariableNode})
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    obs = pgm.observed_values[node.variable]
    is_observed = !isnothing(obs)

    if !is_observed
        cpd = zeros([length(parent.support) for parent in parents]..., length(node.support))
    else
        cpd = zeros([length(parent.support) for parent in parents]...)
    end
    
    for assignment in Iterators.product([1:length(parent.support) for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, i) in zip(parents, assignment)
            X[parent.variable] = parent.support[i]
        end
        d = dist(X)
        if !is_observed
            for (i, x) in enumerate(node.support)
                cpd[assignment..., i] = pdf(d, x) # TODO: replace with logpdf
            end
        else
            x = obs(X)
            cpd[assignment...] = pdf(d, x) # TODO: replace with logpdf
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
        if isnothing(pgm.observed_values[v])
            node.support = get_support(pgm, node, parents)
            cpd = get_table(pgm, node, parents)
            factor_node = FactorNode(push!(parents, node), cpd)
        else
            cpd = get_table(pgm, node, parents)
            factor_node = FactorNode(parents, cpd)
        end
        for neighbour in factor_node.neighbours
            push!(neighbour.neighbours, factor_node)
        end
        push!(factor_nodes, factor_node)
    end
    variable_nodes = variable_nodes[isnothing.(pgm.observed_values)]
    return variable_nodes, factor_nodes
end

function factor_product(A::FactorNode, B::FactorNode)::FactorGraphNode
    a_size = size(A.table)
    b_size = size(B.table)
    common_vars = A.neighbours ∩ B.neighbours
    if isempty(common_vars)
        table = Array{Float64}(undef, a_size..., b_size...)
        # reshaping shares data
        a_table = reshape(A.table, a_size..., ones(Int, length(b_size))...)
        b_table = reshape(B.table, ones(Int, length(a_size))..., b_size...)
        broadcast!(*, table, a_table, b_table)
        vars = vcat(A.neighbours, B.neighbours)
        return FactorNode(vars, table)
    else
        a_common_mask = [v in common_vars for v in A.neighbours]
        b_common_mask = [v in common_vars for v in B.neighbours]
        a_common_ixs = collect(1:length(A.neighbours))[a_common_mask]
        b_common_ixs = collect(1:length(B.neighbours))[b_common_mask]
        common_vars = A.neighbours[a_common_mask] # sorted
        b_ordering = [findfirst(av -> av==bv, common_vars) for bv in B.neighbours[b_common_mask]]
        @assert length(b_ordering) == length(common_vars)
        @assert common_vars[b_ordering] == B.neighbours[b_common_mask]
        @assert a_size[a_common_ixs] == b_size[b_common_ixs]

        # println("common_vars: ", common_vars)
        # println("b_common_vars: ", B.neighbours[b_common_mask])

        a_size_uncommon = a_size[.!a_common_mask]
        b_size_uncommon = b_size[.!b_common_mask]
        
        table = Array{Float64}(undef,  a_size[a_common_ixs]..., a_size_uncommon..., b_size_uncommon...)

        a_selection = Any[Colon() for _ in 1: length(a_size)]
        b_selection = Any[Colon() for _ in 1: length(b_size)]
        table_colon = fill(Colon(), length(a_size_uncommon) + length(b_size_uncommon))
        for common_ixs in Iterators.product([1:length(v.support) for v in common_vars]...)
            a_selection[a_common_mask] .= common_ixs
            b_selection[b_common_mask] .= common_ixs[b_ordering]
            
            a_table = reshape(view(A.table, a_selection...), a_size_uncommon..., ones(Int, length(b_size_uncommon))...)
            b_table = reshape(view(B.table, b_selection...), ones(Int, length(a_size_uncommon))..., b_size_uncommon...)

            broadcast!(*, view(table, common_ixs..., table_colon...), a_table, b_table)
        end

        vars = vcat(common_vars, A.neighbours[.!a_common_mask], B.neighbours[.!b_common_mask])
        # println("vars: ", vars)
        return FactorNode(vars, table)
    end
end

export get_factor_graph, factor_product