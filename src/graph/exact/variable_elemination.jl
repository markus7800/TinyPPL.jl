

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

function variable_elimination(pgm::PGM; marginal_variables=nothing)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    variable_elimination(pgm, variable_nodes, factor_nodes, parse_marginal_variables(pgm, marginal_variables))
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}; marginal_variables=nothing)
    variable_elimination(pgm, variable_nodes, factor_nodes, parse_marginal_variables(pgm, marginal_variables))
end

struct UndirectedEdge
    x::Int
    y::Int
end
function get_undirected_edge(x::Int, y::Int)::UndirectedEdge
    return UndirectedEdge(min(x,y), max(x,y))
end
function min_fill(node::Int, undirected_graph::Set{UndirectedEdge}, node_to_var::Dict{Int, VariableNode}, weighted::Bool=true)::Int
    adjacent_edges = [edge for edge in undirected_graph if edge.x == node || edge.y == node]
    neighbours = [edge.x == node ? edge.y : edge.x for edge in adjacent_edges]

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

function get_elimination_order(pgm::PGM, variable_nodes::Vector{VariableNode}, marginal_variables::Vector{Int})::Vector{VariableNode}
    nodes = Set(node for node in 1:pgm.n_variables if !(node in marginal_variables) && isnothing(pgm.observed_values[node]))
    undirected_graph = Set(get_undirected_edge(x,y) for (x,y) in pgm.edges if (x in nodes) && (y in nodes))

    node_to_var = Dict(var.variable => var for var in variable_nodes)

    ordering = VariableNode[]

    while !isempty(nodes)
        node = argmin(node -> min_fill(node, undirected_graph, node_to_var, true), nodes)
        push!(ordering, node_to_var[node])
        delete!(nodes, node)
        neighbours = [edge.x == node ? edge.y : edge.x for edge in undirected_graph if edge.x == node || edge.y == node]

        for x in neighbours, y in neighbours
            x == y && continue
            edge = get_undirected_edge(x, y)
            push!(undirected_graph, edge)
        end
    end

    return ordering
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}, marginal_variables::Vector{Int})
    elimination_order = get_elimination_order(pgm, variable_nodes, marginal_variables)
    variable_elimination(pgm, factor_nodes, elimination_order)
end

function variable_elimination(pgm::PGM, factor_nodes::Vector{FactorNode}, elimination_order::Vector{VariableNode})
    # println("elimination_order: ", elimination_order)

    factor_nodes = Set(factor_nodes)    
    # println("factor_nodes: ", factor_nodes)
    # println()

    for node in elimination_order
        # println("node to eliminate: ", node)
        neighbour_factors = Set(f for f in factor_nodes if node in f.neighbours)
        # println("neighbour_factors: ", neighbour_factors)

        for f in neighbour_factors
            delete!(factor_nodes, f)
        end

        psi = reduce(factor_product, neighbour_factors)
        tau = factor_sum(psi, [node])
        # println("tau: ", tau)
        push!(factor_nodes, tau)

        # println("factor_nodes: ", factor_nodes)
        # println()
    end

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