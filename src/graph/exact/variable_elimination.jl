
function parse_marginal_variables(pgm::PGM, v::Vector{Int})::Vector{Int}
    return v
end
function parse_marginal_variables(pgm::PGM, addresses::Vector{Any})::Vector{Int}
    addr_to_variable = Dict(addr => i for (i,addr) in enumerate(pgm.addresses))
    return Int[addr_to_variable[addr] for addr in addresses]
end


# eliminates all variables except the marginal_variables.
# marginal_variables default to the variables in the return expression
# otherwise marginal_variables can be givein in terms of addresses or PGM nodes
function variable_elimination(pgm::PGM; marginal_variables=nothing, order::Symbol=:Topological)
    variable_elimination(pgm, variable_nodes, factor_nodes, marginal_variables, order)
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}; marginal_variables=nothing, order::Symbol=:Topological)
    if isnothing(marginal_variables)
        marginal_variables = return_expr_variables(pgm)
    else
        marginal_variables = parse_marginal_variables(pgm, marginal_variables)
    end
    variable_nodes, factor_nodes = get_factor_graph(pgm)

    variable_elimination(pgm, variable_nodes, factor_nodes, marginal_variables, order)
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}, marginal_variables::Vector{Int}, order::Symbol)
    elimination_order = get_elimination_order(pgm, variable_nodes, marginal_variables, order)
    variable_elimination(variable_nodes, elimination_order)
end

# eliminates all variables specified in elimination_order
function variable_elimination(variable_nodes::Vector{VariableNode}, elimination_order::Vector{VariableNode})
    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)

    @progress for node in elimination_order
        neighbour_factors = factor_nodes[node]

        # multiply all neighbouring factors of node
        psi = reduce(factor_product, neighbour_factors)

        # eliminate node
        tau = factor_sum(psi, [node])
        # println(node, ": ", tau)

        # delete all neighbour_factors ...
        for f in neighbour_factors
            for v in f.neighbours
                delete!(factor_nodes[v], f)
            end
        end

        # ... and replace them with new factor tau
        # it is best to elimate variables in the order such that tau is small
        for v in tau.neighbours
            push!(factor_nodes[v], tau)
        end

        # variable successfully eliminated
        delete!(factor_nodes, node)
    end
    
    factor_nodes = reduce(∪, values(factor_nodes))
    return reduce(factor_product, factor_nodes)
end

export variable_elimination