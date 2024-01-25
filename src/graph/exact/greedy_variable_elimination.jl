
mutable struct ReductionSize
    v::VariableNode
    nodes::Set{VariableNode} # set of variables connected to node v via some factor
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

function greedy_variable_elimination(variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int})
    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)

    reduction_size = Dict{VariableNode, ReductionSize}()
    reduction_size_heap = Vector{ReductionSize}(undef, length(variable_nodes) - length(marginal_variables))
    i = 1
    for v in variable_nodes
        (v.variable in marginal_variables) && continue
        r = ReductionSize(v, Set{VariableNode}(), 0, 1, 0, i, 0)
        for f in v.neighbours
            r.individual += length(f.table)
            for n in f.neighbours
                if !(n in r.nodes) && !(n == v)
                    push!(r.nodes, n)
                    r.combined *= length(n.support)
                end
            end
        end
        r.reduction = r.combined - r.individual
        reduction_size[v] = r
        reduction_size_heap[i] = r
        i += 1
        r.metric = get_metric(r)
    end
    heapify!(reduction_size_heap)

    @progress for _ in 1:(length(variable_nodes)-length(marginal_variables))
        r = pop!(reduction_size_heap)
        node = r.v
        # _, r2 = argmin(t -> get_metric(t[2]), reduction_size)
        # @assert (get_metric(r) == get_metric(r2))

        neighbour_factors = factor_nodes[node]

        psi = reduce(factor_product, neighbour_factors)
        tau = factor_sum(psi, [node])
        # println(node, ": ", tau)
        # println("neighbour_factors: ", neighbour_factors)
        # @assert r.reduction == reduction_size_func(node)

        # We add tau to all nodes in tau.neighbours and remove all neighbour_factors f from its neighbours. (1)
        # After the deletion, the set of variables connected to a node via some factor (reduction_size[v].nodes) is decreased by node.
        # Proof:
        # Let v be a variable connected to node via factor f.

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

        for f in neighbour_factors
            for v in f.neighbours
                (v == node) && continue

                # @assert f in factor_nodes[v]
                # @assert f.neighbours ⊆ (tau.neighbours ∪ [node])
                # @assert !(f.neighbours ⊆ tau.neighbours)
                # @assert v in tau.neighbours

                delete!(factor_nodes[v], f)

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
        end

        for v in tau.neighbours
            push!(factor_nodes[v], tau)

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

        delete!(factor_nodes, node)
        delete!(reduction_size, node)
    end
    
    factor_nodes = reduce(∪, values(factor_nodes))
    return reduce(factor_product, factor_nodes)
end


export greedy_variable_elimination
