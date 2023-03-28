
import Distributions: support, logpdf, pdf

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
export VariableNode
function Base.show(io::IO, variable_node::VariableNode)
    print(io, "VariableNode(", (variable_node.variable, variable_node.address), "; ", length(variable_node.support), ")")
end
 
mutable struct FactorNode <: FactorGraphNode
    neighbours::Vector{VariableNode} # variables
    table::Array{Float64}
end
function Base.show(io::IO, factor_node::FactorNode)
    print(io, "FactorNode(", [(n.variable, n.address) for n in factor_node.neighbours], "; ", size(factor_node.table), ")")
    # println(io, factor_node.table)
end
export FactorNode

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
function get_table(pgm::PGM, node::VariableNode, parents::Vector{VariableNode}, logscale::Bool)
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
                cpd[assignment..., i] = logscale ? logpdf(d, x) : pdf(d, x)
            end
        else
            x = obs(X)
            cpd[assignment...] = logscale ? logpdf(d, x) : pdf(d, x)
        end
    end

    return cpd
end

function get_factor_graph(pgm::PGM; logscale::Bool=true, sorted::Bool=true)
    variable_nodes = [VariableNode(i, pgm.addresses[i]) for i in 1:pgm.n_variables]
    factor_nodes = FactorNode[]
    for v in pgm.topological_order
        parents = VariableNode[variable_nodes[x] for (x,y) in pgm.edges if y == v]
        node = variable_nodes[v]
        if isnothing(pgm.observed_values[v])
            node.support = get_support(pgm, node, parents)
            cpd = get_table(pgm, node, parents, logscale)
            factor_node = FactorNode(push!(parents, node), cpd)
        else
            cpd = get_table(pgm, node, parents, logscale)
            factor_node = FactorNode(parents, cpd)
        end
        if sorted # have to sort for ops
            factor_node = factor_permute_vars(factor_node, sort(factor_node.neighbours, lt=(x,y)->x.variable<y.variable))
        end
        for neighbour in factor_node.neighbours
            push!(neighbour.neighbours, factor_node)
        end
        push!(factor_nodes, factor_node)
    end
    # likelihood of observed variables is integrated in factor
    variable_nodes = variable_nodes[isnothing.(pgm.observed_values)]
    return variable_nodes, factor_nodes
end

function factor_product(A::FactorNode, B::FactorNode)::FactorNode
    # @assert issorted([a.variable for a in A.neighbours])
    # @assert issorted([b.variable for b in B.neighbours])
    i = 1
    j = 1
    a_size = Int[]
    b_size = Int[]
    vars = VariableNode[]
    while i <= length(A.neighbours) && j <= length(B.neighbours)
        a = A.neighbours[i]
        b = B.neighbours[j]
        if a.variable == b.variable
            push!(a_size, length(a.support))
            push!(b_size, length(b.support))
            push!(vars, a)
            i += 1
            j += 1
        elseif a.variable < b.variable
            push!(a_size, length(a.support))
            push!(b_size, 1)
            i += 1
            push!(vars, a)
        else # a.variable > b.variable
            push!(a_size, 1)
            push!(b_size, length(b.support))
            j += 1
            push!(vars, b)
        end
    end
    while i <= length(A.neighbours)
        a = A.neighbours[i]
        push!(a_size, length(a.support))
        push!(b_size, 1)
        i += 1
        push!(vars, a)
    end
    while j <= length(B.neighbours)
        b = B.neighbours[j]
        push!(a_size, 1)
        push!(b_size, length(b.support))
        j += 1
        push!(vars, b)
    end
    # @assert length(a_size) == length(b_size)
    # @assert issorted([v.variable for v in vars])
    # @assert Set(vars) == Set(A.neighbours) ∪ Set(B.neighbours)

    a_table = reshape(A.table, Tuple(a_size))
    b_table = reshape(B.table, Tuple(b_size))

    table = broadcast(+, a_table, b_table)

    return FactorNode(vars, table)
end

# A / B
function factor_division(A::FactorNode, B::FactorNode)::FactorNode
    a_size = size(A.table)
    b_size = size(B.table)
    @assert B.neighbours ⊆ A.neighbours

    B = factor_permute_vars(B, [v for v in A.neighbours if v in B.neighbours]) # same factor, but variables are in order of A
    b_table = reshape(B.table, size(B.table)..., ones(Int, length(a_size) - length(b_size))...)
    table = A.table .- b_table # -Inf - -Inf = -Inf <-> 0/0 = 0
    @assert all(table .<= 0)

    return FactorNode(A.neighbours, table)
end

function factor_sum(factor_node::FactorNode, dims::Vector{Int})::FactorNode
    variables = [v for (i,v) in enumerate(factor_node.neighbours) if !(i in dims)]
    size = [length(v.support) for v in variables]
    # table = mapslices(sum, factor_node.table, dims=dims)
    # table = mapslices(logsumexp, factor_node.table, dims=dims)
    table = log.(sum(exp, factor_node.table, dims=dims))
    table = reshape(table, size...)
    return FactorNode(variables, table)
end


function factor_sum(factor_node::FactorNode, variables::Vector{VariableNode})::FactorNode
    dims = [i for (i,v) in enumerate(factor_node.neighbours) if v in variables]
    return factor_sum(factor_node, dims)
end

function factor_permute_vars(factor_node::FactorNode, perm::Vector{Int})::FactorNode
    return FactorNode(factor_node.neighbours[perm], permutedims(factor_node.table, perm))
end

function factor_permute_vars(factor_node::FactorNode, variables::Vector{VariableNode})::FactorNode
    perm = Int[findfirst(av -> av==v, factor_node.neighbours) for v in variables]
    permuted_factor = factor_permute_vars(factor_node, perm)
    @assert variables == permuted_factor.neighbours
    return permuted_factor
end

export get_factor_graph, factor_product, factor_sum, factor_permute_vars