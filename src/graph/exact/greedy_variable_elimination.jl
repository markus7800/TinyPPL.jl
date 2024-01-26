# Greedy Variable Elimination works by eliminating the variable that
# reduces the cumulative size of all factors the most.
# We find this node by each variable a ReductionSize struct in a heap,
# which we update on the fly to be performant.

mutable struct ReductionSize
    # node to eliminate
    v::VariableNode
    # set of variables connected to node v via some factor,
    # if we eliminate v, tau will be a factor of these nodes
    nodes::Set{VariableNode}

    # metrics
    individual::Int # sum of sizes of all factors connected to node v
    combined::Int # sum of size of factor resulting from multiplying all factors connected to node v and summing out v
    reduction::Int # individual - combined

    position::Int # position in heap
    metric::Int # cached metric
end
function get_metric(r::ReductionSize)
    r.combined # r.reduction # combined works better
end

function Base.isless(a::ReductionSize, b::ReductionSize)
    a.metric < b.metric
end

heap_left(i::Int) = 2*i
heap_right(i::Int) = 2*i + 1
heap_parent(i::Int) = div(i, 2)

# stores value x at position i, if needed, moves down the inserted value to maintain heap
function heapify_down!(A::Vector{ReductionSize}, i::Int, x::ReductionSize=A[i], n::Int=length(A))::Int
    while (l = heap_left(i)) ≤ n
        j = (r = heap_right(i)) > n || isless(A[l], A[r]) ? l : r
        isless(A[j], x) || break
        A[i] = A[j]
        A[i].position = i
        i = j
    end
    A[i] = x
    x.position = i
    return i
end

# stores value x at position i, if needed, moves up the inserted value to maintain heap
function heapify_up!(A::Vector{ReductionSize}, i::Int, x::ReductionSize=A[i])::Int
    while (j = heap_parent(i)) ≥ 1 && isless(x, A[j])
        A[i] = A[j] # move parent down
        A[i].position = i
        i = j
    end
    A[i] = x
    x.position = i
    return i
end

function heapify!(A::Vector{ReductionSize})
    n = length(A)
    for i in heap_parent(n):-1:1
        heapify_down!(A, i, A[i], n)
    end
end
function isheap(A::Vector{ReductionSize})
    n = length(A)
    for i in 1:div(n, 2)
        l = heap_left(i)
        r = heap_right(i)
        if isless(A[l], A[i]) || (r ≤ n && isless(A[r], A[i]))
            return false
        end
    end
return true
end

function Base.pop!(A::Vector{ReductionSize})
    n = length(A)
    x = A[1]
    if n > 1
        # Peek the last value and down-heapify starting at the root of the binary heap to insert it.
        y = A[n]
        heapify_down!(A, 1, y, n - 1)
    end
    resize!(A, n - 1)
    return x
end

function initialise_reduction_size_heap(variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int})
    reduction_size = Dict{VariableNode, ReductionSize}()
    reduction_size_heap = Vector{ReductionSize}(undef, length(variable_nodes) - length(marginal_variables))
    i = 1
    for v in variable_nodes
        (v.variable in marginal_variables) && continue
        r = ReductionSize(v, Set{VariableNode}(), 0, 1, 0, i, 0)
        for f in v.neighbours
            # sum of size of all neighbouring factors
            r.individual += length(f.table)

            # if we eliminate v, tau will be a factor of with scope r.nodes
            # r.nodes are all variables that are connected to v via some neighbours factor
            # the size of tau is the product of the support of all r.nodes
            # as v is eliminated it does not contribute to the size of tau.
            for n in f.neighbours
                if !(n in r.nodes) && !(n == v)
                    push!(r.nodes, n)
                    r.combined *= length(n.support)
                end
            end
        end
        # we replace all neighbouring factors with one big factor tau
        # the change in total factor sizes is r.reduction
        r.reduction = r.combined - r.individual

        reduction_size[v] = r
        reduction_size_heap[i] = r
        i += 1

        # we choose a metric calculated from r.reduction, r.combined, r.individual
        r.metric = get_metric(r)
    end
    
    # make heap 
    heapify!(reduction_size_heap)

    return reduction_size, reduction_size_heap
end

function remove_f_from_reduction_score_of_v!(reduction_size::Dict{VariableNode, ReductionSize}, node::VariableNode, v::VariableNode, f::FactorNode)
    if haskey(reduction_size, v) # v is not a marginal variable
        r = reduction_size[v]
        # @assert f.neighbours ⊆ r.nodes ∪ [node, v]
        
        r.individual -= length(f.table)
        if node in r.nodes
            # node can be in multiple f
            r.combined /= length(node.support)
            delete!(r.nodes, node)
        end
    end
end

function add_tau_to_reduction_score_of_v_and_heapify!(
    reduction_size::Dict{VariableNode, ReductionSize}, reduction_size_heap::Vector{ReductionSize},
    v::VariableNode, tau::FactorNode)

    if haskey(reduction_size, v) # v is not a marginal variable
        r = reduction_size[v]
        r.individual += length(tau.table)
        for n in tau.neighbours
            if !(n in r.nodes) && !(n == v)
                push!(r.nodes, n)
                r.combined *= length(n.support)
            end
        end
        r.reduction = r.combined - r.individual

        before = r.metric
        after = get_metric(r)
        r.metric = after

        # resort
        if before < after
            heapify_down!(reduction_size_heap, r.position)
        else
            heapify_up!(reduction_size_heap, r.position)
        end
        # @assert isheap(reduction_size_heap)
    end
end

function get_greedy_elimination_order(variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int})
    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)

    # we make heap such that we can simply pop the best node to eliminate
    reduction_size, reduction_size_heap = initialise_reduction_size_heap(variable_nodes, marginal_variables)

    elimination_order = VariableNode[]
    @progress for _ in 1:(length(variable_nodes)-length(marginal_variables))
        r = pop!(reduction_size_heap)
        node = r.v
        push!(elimination_order, node)
        # _, r2 = argmin(t -> get_metric(t[2]), reduction_size)
        # @assert (get_metric(r) == get_metric(r2))

        neighbour_factors = factor_nodes[node]

        # multiply all neighbouring factors of node
        # psi = reduce(factor_product, neighbour_factors)
        # eliminate node
        # tau = factor_sum(psi, [node])

        # mock the computation
        tau_neighbours = reduce(∪, Set(f.neighbours) for f in neighbour_factors; init=Set{VariableNode}())
        delete!(tau_neighbours, node)
        tau = FactorNode(sort!(collect(tau_neighbours)), Float64[])
        
        # println(node, ": ", tau)
        # println("neighbour_factors: ", neighbour_factors)
        # @assert r.reduction == reduction_size_func(node)

        # We add tau to all nodes in tau.neighbours and remove all neighbour_factors f from its neighbours. (1)
        # After the deletion, the set of variables connected to node via some factor (reduction_size[v].nodes) is decreased by node.
        # Proof:
        # Let v be a variable connected to `node` via factor `f`.

        # f is in neighbour_factors
        # v is in f.neighbours
        # v is in tau.neighbours
        # reduction_size[v].nodes = ∪ f.neighbours for f in factor_nodes[v] \ [v]

        # For all factors g neighbouring v (g ∈ factor_nodes[v]), if node in g.neighbours => g in neighbour_factors
        # => g is removed from factor_nodes[v], therefore eliminating node from reduction_size[v].nodes (there is atleast f).

        # Let w be a variable removed from reduction_size[v].nodes with (1).
        # => There has to be a factor f in factor_nodes[v] with w ∈ f.neighbours that is removed => f ∈ neighbour_factors
        # tau.neighbours = (∪ f.neighbours for f in neighbour_factors) \ [node]
        # => f.neighbours \ [node] ⊆ tau.neighbours
        # tau is added to factor_nodes[v] => tau.neighbours ⊆ reduction_size[v].nodes ∪ [v]
        # w ∈ f.neighbours and w ∉ reduction_size[v].nodes ∪ [v] and f.neighbours \ [node] ⊆ reduction_size[v].nodes ∪ [v] implies that w == node

        # delete all neighbour_factors ...
        for f in neighbour_factors
            for v in f.neighbours
                (v == node) && continue
                delete!(factor_nodes[v], f)

                # @assert f in factor_nodes[v]
                # @assert f.neighbours ⊆ (tau.neighbours ∪ [node])
                # @assert !(f.neighbours ⊆ tau.neighbours)
                # @assert v in tau.neighbours

                # variables of neighbouring factors can be in other factors
                # we have to update their reduction score
                # if we elimate them in the future there is not the factor `f` to consider anymore
                # heap will be updated later for all v in tau.neighbours:  ∪ [v in f.neighbours] ⊆ tau.neighbours ∪ [node]
                remove_f_from_reduction_score_of_v!(reduction_size, node, v, f)
            end
        end

        for v in tau.neighbours
            push!(factor_nodes[v], tau)
            add_tau_to_reduction_score_of_v_and_heapify!(reduction_size, reduction_size_heap, v, tau)
        end

        delete!(factor_nodes, node)
        delete!(reduction_size, node)
    end
    
    return elimination_order
end
export get_greedy_elimination_order

# this is equivalent to variable_elimination(variable_nodes, get_greedy_elimination_order(variable_nodes, marginal_variables))
# but book-keeping variable and factor nodes has to be done only once.
# This is usually the fastest variable elimination algorithm.
function greedy_variable_elimination(variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int})
    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)

    # we make heap such that we can simply pop the best node to eliminate
    reduction_size, reduction_size_heap = initialise_reduction_size_heap(variable_nodes, marginal_variables)

    @progress for _ in 1:(length(variable_nodes)-length(marginal_variables))
        r = pop!(reduction_size_heap)
        node = r.v

        neighbour_factors = factor_nodes[node]

        # multiply all neighbouring factors of node
        psi = reduce(factor_product, neighbour_factors)
        # eliminate node
        tau = factor_sum(psi, [node])
        
        # delete all neighbour_factors ...
        for f in neighbour_factors
            for v in f.neighbours
                (v == node) && continue
                delete!(factor_nodes[v], f)
                remove_f_from_reduction_score_of_v!(reduction_size, node, v, f)
            end
        end

        for v in tau.neighbours
            push!(factor_nodes[v], tau)
            add_tau_to_reduction_score_of_v_and_heapify!(reduction_size, reduction_size_heap, v, tau)
        end

        delete!(factor_nodes, node)
        delete!(reduction_size, node)
    end
    
    factor_nodes = reduce(∪, values(factor_nodes))
    return reduce(factor_product, factor_nodes)
end

function greedy_variable_elimination(pgm::PGM; marginal_variables=nothing)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    greedy_variable_elimination(pgm, variable_nodes, factor_nodes, marginal_variables=marginal_variables)
end

function greedy_variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}; marginal_variables=nothing)
    if isnothing(marginal_variables)
        marginal_variables = return_expr_variables(pgm)
    else
        marginal_variables = parse_marginal_variables(pgm, marginal_variables)
    end
    greedy_variable_elimination(variable_nodes, marginal_variables)
end

export greedy_variable_elimination