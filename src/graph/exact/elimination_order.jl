
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

    # this is slow for large graphs
    
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