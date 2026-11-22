
# cost of eliminating node = number of fill edges (edges between neighbours of node that are not yet in the graph)
# weighted: each fill edge costs the product of the support sizes of its endpoints
function min_fill(node::Int, node_to_var::Dict{Int, VariableNode}, node_to_neighbours::Dict{Int,Set{Int}}, weighted::Bool=true)::Int
    neighbours = collect(node_to_neighbours[node])
    cost = 0
    for (i, x) in enumerate(neighbours), y in neighbours[i+1:end]
        if !(y in node_to_neighbours[x])
            if weighted
                cost += length(node_to_var[x].support) * length(node_to_var[y].support)
            else
                cost += 1
            end
        end
    end
    return cost
end
# cost of eliminating node = number of neighbours (scope of the factor tau created by eliminating node)
function min_neighbour(node::Int, node_to_neighbours::Dict{Int,Set{Int}})
    return length(node_to_neighbours[node])
end

function get_topological_elimination_order(pgm::PGM, variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int})
    var_to_node = Dict(v.variable => v for v in variable_nodes);
    ordering = [var_to_node[v] for v in pgm.topological_order if !(v in marginal_variables) && haskey(var_to_node, v)]
    return ordering
end

# PGM 9.4.3.2 Greedy Search with cost criteria :MinNeighbours, :MinFill, :WeightedMinFill
# Works on the interaction graph (moralised graph) of the factor graph: two variables are connected if they share a factor.
# (this includes connections between parents of observed variables and to marginal_variables, which are not eliminated)
# Eliminating a node connects all its neighbours (fill edges) and removes it from the graph.
function get_min_neighbour_fill_elimination_order(pgm::PGM, variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int}, order::Symbol)
    @assert order in (:MinNeighbours, :MinFill, :WeightedMinFill) "Unsupported cost $order."

    # interaction graph
    node_to_neighbours = Dict{Int,Set{Int}}(v.variable => Set{Int}() for v in variable_nodes)
    for v in variable_nodes
        for factor in v.neighbours
            for w in factor.neighbours
                w.variable == v.variable && continue
                push!(node_to_neighbours[v.variable], w.variable)
            end
        end
    end

    node_to_var = Dict(var.variable => var for var in variable_nodes)

    # this is slow for large graphs, we could also rewrite this with heaps / priority queues
    nodes = Set(v.variable for v in variable_nodes if !(v.variable in marginal_variables))
    ordering = VariableNode[]

    @progress for _ in 1:length(nodes)
        if order == :WeightedMinFill
            node = argmin(node -> min_fill(node, node_to_var, node_to_neighbours, true), nodes)
        elseif order == :MinFill
            node = argmin(node -> min_fill(node, node_to_var, node_to_neighbours, false), nodes)
        else # order == :MinNeighbours
            node = argmin(node -> min_neighbour(node, node_to_neighbours), nodes)
        end

        push!(ordering, node_to_var[node])
        delete!(nodes, node)

        # eliminate node: connect all neighbours and remove node from graph
        neighbours = pop!(node_to_neighbours, node)
        for x in neighbours
            delete!(node_to_neighbours[x], node)
            union!(node_to_neighbours[x], neighbours)
            delete!(node_to_neighbours[x], x)
        end
    end
    @assert isempty(nodes)

    return ordering
end

function get_elimination_order(pgm::PGM, variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int}, order::Symbol)::Vector{VariableNode}
    if order == :Topological
        return get_topological_elimination_order(pgm, variable_nodes, marginal_variables)
    elseif order == :Greedy
        return get_greedy_elimination_order(variable_nodes, marginal_variables)
    elseif order in (:MinFill, :WeightedMinFill, :MinNeighbours)
        return get_min_neighbour_fill_elimination_order(pgm, variable_nodes, marginal_variables, order)
    else
        error("Unsupported elimnation order $order.")
    end
    
end
export get_elimination_order