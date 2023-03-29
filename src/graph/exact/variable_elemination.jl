

function return_expr_variables(pgm::PGM)::Vector{Int}
    return Int[pgm.sym_to_ix[sym] for sym in get_free_variables(pgm.symbolic_return_expr) ∩ keys(pgm.sym_to_ix)]
end
export return_expr_variables

function parse_marginal_variables(pgm::PGM, ::Nothing)::Vector{Int}
    return return_expr_variables(pgm)
end
function parse_marginal_variables(pgm::PGM, v::Vector{Int})::Vector{Int}
    return v
end
function parse_marginal_variables(pgm::PGM, addresses::Vector{Any})::Vector{Int}
    addr_to_variable = Dict(addr => i for (i,addr) in enumerate(pgm.addresses))
    return Int[addr_to_variable[addr] for addr in addresses]
end

function variable_elimination(pgm::PGM; marginal_variables=nothing, order::Symbol=:Topological)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    variable_elimination(pgm, variable_nodes, factor_nodes, parse_marginal_variables(pgm, marginal_variables), order)
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}; marginal_variables=nothing, order::Symbol=:Topological)
    variable_elimination(pgm, variable_nodes, factor_nodes, parse_marginal_variables(pgm, marginal_variables), order)
end

struct UndirectedEdge
    x::Int
    y::Int
end
function get_undirected_edge(x::Int, y::Int)::UndirectedEdge
    return UndirectedEdge(min(x,y), max(x,y))
end
function min_fill(node::Int, undirected_graph::Set{UndirectedEdge}, node_to_var::Dict{Int, VariableNode}, node_to_neighbours::Dict{Int,Set{Int}}, weighted::Bool=true)::Int
    if !haskey(node_to_neighbours, node)
        return 0
    end
    neighbours = node_to_neighbours[node]
    cost = 0
    added_edges = Set{UndirectedEdge}()
    for x in neighbours, y in neighbours
        x == y && continue
        edge = get_undirected_edge(x, y)
        if !(edge in undirected_graph) && !(edge in added_edges)
            push!(added_edges, edge)
            if weighted
                cost += length(node_to_var[edge.x].support) * length(node_to_var[edge.y].support)
            else
                cost += 1
            end
        end
    end

    return cost
end
function min_neighbour(node::Int, node_to_neighbours::Dict{Int,Set{Int}})
    if !haskey(node_to_neighbours, node)
        return 0
    end
    return length(node_to_neighbours[node])
end

function get_elimination_order(pgm::PGM, variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int}, order::Symbol)::Vector{VariableNode}
    if order == :Topological
        var_to_node = Dict(v.variable => v for v in variable_nodes);
        ordering = [var_to_node[v] for v in pgm.topological_order if !(v in marginal_variables) && haskey(var_to_node, v)]
        return ordering
    end

    nodes = Set(node for node in 1:pgm.n_variables if !(node in marginal_variables) && isnothing(pgm.observed_values[node]))
    
    undirected_graph = Set(get_undirected_edge(x,y) for (x,y) in pgm.edges if (x in nodes) && (y in nodes))
    if order == :MinNeighbours
        # moralize
        for node in nodes
            parents = [x for (x,y) in pgm.edges if y == node]
            for p1 in parents, p2 in parents
                p1 == p2 && continue
                push!(undirected_graph, get_undirected_edge(p1,p2))
            end
        end
    end
    node_to_neighbours = Dict{Int,Set{Int}}()
    for edge in undirected_graph
        x_neighbours = get!(node_to_neighbours, edge.x, Set{Int}())
        push!(x_neighbours, edge.y)
        y_neighbours = get!(node_to_neighbours, edge.y, Set{Int}())
        push!(y_neighbours, edge.x)
    end

    node_to_var = Dict(var.variable => var for var in variable_nodes)

    ordering = VariableNode[]

    @progress for _ in 1:length(nodes)
        if order == :WeightedMinFill
            node = argmin(node -> min_fill(node, undirected_graph, node_to_var, node_to_neighbours, true), nodes)
        elseif order == :MinFill
            node = argmin(node -> min_fill(node, undirected_graph, node_to_var, node_to_neighbours, false), nodes)
        elseif order == :MinNeighbours
            node = argmin(node -> min_neighbour(node, node_to_neighbours), nodes)
        else
            error("Unsupported cost $cost.")
        end

        push!(ordering, node_to_var[node])
        delete!(nodes, node)
        neighbours = [edge.x == node ? edge.y : edge.x for edge in undirected_graph if edge.x == node || edge.y == node]

        if order == :WeightedMinFill || order == :MinFill
            for x in neighbours, y in neighbours
                x == y && continue
                edge = get_undirected_edge(x, y)
                push!(undirected_graph, edge)
            end
        end
    end
    @assert isempty(nodes)

    return ordering
end
export get_elimination_order

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}, marginal_variables::Vector{Int}, order::Symbol)
    elimination_order = get_elimination_order(pgm, variable_nodes, marginal_variables, order)
    variable_elimination(variable_nodes, elimination_order)
end

function variable_elimination(variable_nodes::Vector{VariableNode}, elimination_order::Vector{VariableNode})
    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)

    @progress for node in elimination_order
        neighbour_factors = factor_nodes[node]

        psi = reduce(factor_product, neighbour_factors)
        tau = factor_sum(psi, [node])
        # println(node, ": ", tau)

        for f in neighbour_factors
            for v in f.neighbours
                delete!(factor_nodes[v], f)
            end
        end

        for v in tau.neighbours
            push!(factor_nodes[v], tau)
        end

        delete!(factor_nodes, node)
    end
    
    factor_nodes = reduce(∪, values(factor_nodes))
    return reduce(factor_product, factor_nodes)
end

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


export variable_elimination, greedy_variable_elimination
