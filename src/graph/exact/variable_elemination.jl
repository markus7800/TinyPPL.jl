

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

        if order == WeightedMinFill || order == :MinFill
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
        # println(size(tau.table))

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

export variable_elimination

function evaluate_return_expr_over_factor(pgm::PGM, factor::FactorNode)
    result = Array{Tuple{Any, Float64}}(undef, size(factor.table))

    X = Vector{Float64}(undef, pgm.n_variables)
    Z = sum(exp, factor.table)
    for indices in Iterators.product([1:length(node.support) for node in factor.neighbours]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (node, i) in zip(factor.neighbours, indices)
            X[node.variable] = node.support[i]
        end
        
        retval = pgm.return_expr(X) # observed values in return expr are subsituted with their value
        prob = exp(factor.table[indices...]) / Z
        result[indices...] = (retval, prob)
    end
    result = reshape(result, :)
    values = Dict(val => 0. for (val, prob) in result)
    for (val, prob) in result
        values[val] = values[val] + prob
    end
    simplified_result = sort([(val, prob) for (val, prob) in values])

    return simplified_result
end

export evaluate_return_expr_over_factor