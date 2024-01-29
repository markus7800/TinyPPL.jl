
import Distributions: support, logpdf, pdf

abstract type FactorGraphNode end 
# in a PGM a variable node wraps one PGM latent variable x
# if there is an edge v -> x or x -> v then there is at least one factor f in neighbours with {x,y} ⊆ f.neighbours
# the support of the variable are all values it can take
mutable struct VariableNode <: FactorGraphNode
    variable::Int # the PGM variable
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
Base.isless(x::VariableNode, y::VariableNode) = x.variable < y.variable
 
# in a PGM a FactorNode corresponds to one CPD
# p(x | pa(x)) in case of a sample statement (neighbours are pa(x) ∪ {x})
# p(y=value | pa(y)) inc ase of an observe statement (neighbours are pa(y))
mutable struct FactorNode <: FactorGraphNode
    neighbours::Vector{VariableNode} # variables
    table::Array{Float64}
    function FactorNode(neighbours::Vector{VariableNode}, table::Array{Float64})::FactorNode
        if !issorted(neighbours)
            perm = sortperm(neighbours)
            neighbours = neighbours[perm]
            table = permutedims(table, perm)
        end
        return new(neighbours, table)
    end

    function FactorNode(neighbours::Vector{VariableNode}, number::Float64)::FactorNode
        @assert isempty(neighbours)
        table = Array{Float64,0}(undef)
        table[] = number
        return new(neighbours, table)
    end
end

function Base.show(io::IO, factor_node::FactorNode)
    print(io, "FactorNode(", [(n.variable, n.address) for n in factor_node.neighbours], "; ", size(factor_node.table), ")")
    # println(io, factor_node.table)
end

Base.similar(factor_node::FactorNode) = FactorNode(factor_node.neighbours, similar(factor_node.table))
export FactorNode

# support is calculated by iterating ofver the product of all parent supports
# thus, parents must already have support
function get_support(pgm::PGM, node::VariableNode, parents::Vector{VariableNode})
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    is_observed = isobserved(pgm, node.variable)

    supp = Set{Float64}()
    # iterate over the product of all parent supports
    for assignment in Iterators.product([parent.support for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, value) in zip(parents, assignment)
            X[parent.variable] = value
        end
        if is_observed
            v = get_observed_value(pgm, node.variable)
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
    is_observed = isobserved(pgm, node.variable)

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
            x = get_observed_value(pgm, node.variable)
            cpd[assignment...] = logscale ? logpdf(d, x) : pdf(d, x)
        end
    end

    return cpd
end

function get_factor_graph(pgm::PGM; logscale::Bool=true)
    # create a variable node for each PGM variable
    variable_nodes = [VariableNode(i, pgm.addresses[i]) for i in 1:pgm.n_variables]
    factor_nodes = FactorNode[]
    for v in pgm.topological_order
        # get all parent variabel nodes pa(v)
        parents = VariableNode[variable_nodes[x] for (x,y) in pgm.edges if y == v]
        node = variable_nodes[v]
        if !isobserved(pgm, v)
            node.support = get_support(pgm, node, parents)
            # create factor node that represents CPD p(v | pa(v))
            cpd = get_table(pgm, node, parents, logscale)
            # factor consists of parents ∪ {node}
            factor_node = FactorNode(push!(parents, node), cpd)
        else
            # each observation is represented by one factor  p(observed_value | pa(v))
            cpd = get_table(pgm, node, parents, logscale)
            # factor consists only of parents
            factor_node = FactorNode(parents, cpd)
        end
        # connect variable nodes to factor node
        for neighbour in factor_node.neighbours
            push!(neighbour.neighbours, factor_node)
        end
        push!(factor_nodes, factor_node)
    end
    # likelihood of observed variables is integrated in factor
    variable_nodes = variable_nodes[1:pgm.n_latents]

    return variable_nodes, factor_nodes
end

# PGM Definition 4.2, Algorithm 10.A.1
# A(X,Y) * B(Y,Z) = C(X,Y,Z)
function factor_product(A::FactorNode, B::FactorNode)::FactorNode
    # @assert issorted([a.variable for a in A.neighbours])
    # @assert issorted([b.variable for b in B.neighbours])
    i = 1
    j = 1
    a_size = Int[]
    b_size = Int[]
    vars = VariableNode[]
    # since neighbours are sorted we can do a "merge sort" like calculation of the final factor size
    # we then use broadcasting to compute factor
    # e.g. for A(X,Y) l x m  and  B(Y,Z) m x n
    # vars = [X,Y,Z] and C(X,Y,Z) l x m x n
    # a_size = (l, m, 1), b_size = (1, m, n)
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

    # reshape according to computed sizes and make use of broadcasting
    a_table = reshape(A.table, Tuple(a_size))
    b_table = reshape(B.table, Tuple(b_size))

    table = broadcast(+, a_table, b_table)

    return FactorNode(vars, table)
end

# PGM (Definition 10.7)
# !if! A * B = C, this factor_division returns C \ A = B
#   A   *   B   =    C
# [X,Y] * [Y,Z] = [X,Y,Z]
# If you are not sure (do not want to compute) which variables are broadcasted
# in A * B, then you can always choose B = similar(C), but C will be constant
# in the dimensions corresponding to variables of A that are not in B.
function factor_division!(C::FactorNode, A::FactorNode, B::FactorNode)::FactorNode
    @assert Set(A.neighbours) ∪ Set(B.neighbours) == Set(C.neighbours)

    a_size = Int[]
    table_sel = []
    for (i,v) in enumerate(C.neighbours)
        if !(v in B.neighbours)
            # X was broadcasted from A we can select it
            push!(table_sel, 1)
        else
            # Y
            push!(table_sel, Colon())
        end
        if v in A.neighbours
            # X, Y
            push!(a_size, length(v.support))
        else
            # Z
            push!(a_size, 1)
        end
    end
    # reshape A to match C
    a_table = reshape(A.table, Tuple(a_size))

    # select away broadcasted dimensions from A
    # B[:,:] = C[1,:,:] - reshape(A)[1,:,:], where reshape(A) is length(X) x length(Y) x 1
    B.table .= view(C.table, table_sel...) .- view(a_table, table_sel...) # -Inf - -Inf = -Inf <-> 0/0 = 0

    return B
end

# X = VariableNode(1,:X); X.support = [1.,2.];
# Y = VariableNode(2,:Y); Y.support = [1.,2.,3.];
# Z = VariableNode(3,:Z); Z.support = [1.,2.,3.,4];
# A = FactorNode([X, Y], -rand(2,3))
# B = FactorNode([Y, Z], -rand(3,4))
# C = factor_product(A,B)

# B2 = factor_division!(C, A, FactorNode(B.neighbours, similar(B.table)))
# B2.table ≈ B.table

# B2 = factor_division!(C, A, FactorNode(C.neighbours, similar(C.table)))
# B2.table[1,:,:] ≈ B.table
# B2.table[2,:,:] ≈ B.table

# A2 = factor_division!(C, B, FactorNode(A.neighbours, similar(A.table)))
# A2.table ≈ A.table

# A2 = factor_division!(C, B, FactorNode(C.neighbours, similar(C.table)))
# A2.table[:,:,1] ≈ A.table
# A2.table[:,:,2] ≈ A.table
# A2.table[:,:,3] ≈ A.table
# A2.table[:,:,4] ≈ A.table

Base.reshape(f::Float64, ::Tuple{}) = f
 
# PGM 9.3.1.1 Factor Marginalisation
# sums out dims from factor
# factor table is in log-space thus the operation is logsumexp
function factor_sum(factor_node::FactorNode, dims::Vector{Int})::FactorNode
    variables = [v for (i,v) in enumerate(factor_node.neighbours) if !(i in dims)]
    size = Tuple([length(v.support) for v in variables])
    # table = mapslices(sum, factor_node.table, dims=dims)
    # table = mapslices(logsumexp, factor_node.table, dims=dims)
    table = log.(sum(exp, factor_node.table, dims=dims))
    table = reshape(table, size)
    return FactorNode(variables, table)
end

# sums out variables from factor_node
# resulting factor has variables factor_node.neighbours \ variables
function factor_sum(factor_node::FactorNode, variables::Vector{VariableNode})::FactorNode
    @assert variables ⊆ factor_node.neighbours
    dims = [i for (i,v) in enumerate(factor_node.neighbours) if v in variables]
    return factor_sum(factor_node, dims)
end

export get_factor_graph, factor_product, factor_division, factor_sum


# returns all variables the return expression depends on
function return_expr_variables(pgm::PGM)::Vector{Int}
    return Int[pgm.sym_to_ix[sym] for sym in get_free_variables(pgm.symbolic_return_expr) ∩ keys(pgm.sym_to_ix)]
end
export return_expr_variables

function add_return_factor!(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    variable_to_node = Dict(node.variable=>node for node in variable_nodes)
    return_variables = [variable_to_node[v] for v in return_expr_variables(pgm)]
    return add_return_factor!(factor_nodes, return_variables)
end

# create a factor which holds all return_variables and initialise it with 0 table
# connect the return factor to the factor_nodes
function add_return_factor!(factor_nodes::Vector{FactorNode}, return_variables::Vector{VariableNode})
    sort!(return_variables)
    return_factor = FactorNode(return_variables, zeros(Tuple([length(node.support) for node in return_variables])))
    for variable in return_variables
        push!(variable.neighbours, return_factor)
    end
    push!(factor_nodes, return_factor)
    return return_factor
end

function evaluate_return_expr_over_factor(pgm::PGM, factor::FactorNode)
    result = Array{Tuple{Any, Float64}}(undef, size(factor.table))

    # normalising constant
    Z = sum(exp, factor.table)

    X = Vector{Float64}(undef, pgm.n_variables)
    # iterate over the product of the support of all factor variables
    for indices in Iterators.product([1:length(node.support) for node in factor.neighbours]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (node, i) in zip(factor.neighbours, indices)
            X[node.variable] = node.support[i]
        end
        
        retval = get_retval(pgm, X) # observed values in return expr are subsituted with their value
        prob = exp(factor.table[indices...]) / Z
        result[indices...] = (retval, prob)  # add return value and its probability to the result
    end
    # transform to list
    result = reshape(result, :)

    # get unique return values and sum up their probability
    values = Dict(val => 0. for (val, prob) in result)
    for (val, prob) in result
        values[val] = values[val] + prob
    end

    # sort the result by value
    simplified_result = sort([(val, prob) for (val, prob) in values])

    return simplified_result
end

export evaluate_return_expr_over_factor