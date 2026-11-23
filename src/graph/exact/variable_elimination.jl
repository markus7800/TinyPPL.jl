
function parse_variables(pgm::PGM, v::Vector{Int})::Vector{Int}
    return v
end
function parse_variables(pgm::PGM, vars::Vector{VariableNode})::Vector{Int}
    return Int[v.variable for v in vars]
end
function parse_variables(pgm::PGM, addresses::Vector{Any})::Vector{Int}
    addr_to_variable = Dict(addr => i for (i,addr) in enumerate(pgm.addresses))
    return Int[addr_to_variable[addr] for addr in addresses]
end
function parse_variables(pgm::PGM, addresses::Vector{Symbol})::Vector{Int}
    return parse_variables(pgm, convert(Vector{Any}, addresses))
end

export parse_variables

function get_joint_factor(variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    res = reduce(factor_product, factor_nodes)
    log_evidence = logsumexp(res.table)
    return res, exp(log_evidence)
end
function get_joint_factor(pgm::PGM)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return get_joint_factor(variable_nodes, factor_nodes)
end

export get_joint_factor

# eliminates all variables except the marginal_variables.
# marginal_variables default to the variables in the return expression
# otherwise marginal_variables can be givein in terms of addresses or PGM nodes
function variable_elimination(pgm::PGM; marginal_variables=nothing, order::Symbol=:Topological)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    variable_elimination(pgm, variable_nodes, factor_nodes, marginal_variables=marginal_variables, order=order)
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}; marginal_variables=nothing, order::Symbol=:Topological)
    if isnothing(marginal_variables)
        marginal_variables = return_expr_variables(pgm)
    else
        marginal_variables = parse_variables(pgm, marginal_variables)
    end
    return variable_elimination(pgm, variable_nodes, factor_nodes, marginal_variables, order)
end

function variable_elimination(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}, marginal_variables::Vector{Int}, order::Symbol)
    elimination_order = get_elimination_order(pgm, variable_nodes, marginal_variables, order)
    return variable_elimination(variable_nodes, elimination_order; scalar_factors=get_scalar_factors(factor_nodes))
end

struct VariableEliminationResult
    factor::FactorNode # unnormalised log factor over marginal variables
    evidence::Float64 # may underflow, use log_evidence
    log_evidence::Float64
end

"""
Result convention of exact inference algorithms (variable_elimination, greedy_variable_elimination,
belief_propagation, junction_tree_message_passing): the result has fields `log_evidence` (and `evidence`, which may underflow)
and `get_posterior(res)` returns the normalised posterior table (linear space) over the query variables:
the marginal variables for variable elimination and the scope of the root factor for belief propagation / junction trees
(the return variables with `return_factor_as_root=true`).
Calibrated belief propagation / junction tree results also provide the marginals of all variables with `get_marginals(res)`.
"""
get_posterior(res::VariableEliminationResult) = exp.(res.factor.table .- res.log_evidence)
export get_posterior

# factors with empty scope (e.g. observations that do not depend on latent variables)
# they are not neighbours of any variable node
get_scalar_factors(factor_nodes::Vector{FactorNode}) = FactorNode[f for f in factor_nodes if isempty(f.neighbours)]

# multiplies the factors that are left after variable elimination
# scalar_factors are factors with empty scope, which are not connected to any variable (e.g. if a connected component is eliminated)
function get_variable_elimination_result(factor_nodes::Dict{VariableNode,<:Set}, scalar_factors::Vector{FactorNode})
    remaining_factors = reduce(∪, values(factor_nodes); init=Set{FactorNode}())::Set
    res = reduce(factor_product, remaining_factors ∪ scalar_factors; init=FactorNode(VariableNode[], 0.))
    log_evidence = logsumexp(res.table)
    return VariableEliminationResult(res, exp(log_evidence), log_evidence)
end

# eliminates all variables specified in elimination_order
function variable_elimination(variable_nodes::Vector{VariableNode}, elimination_order::Vector{VariableNode}; scalar_factors::Vector{FactorNode}=FactorNode[])
    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)
    scalar_factors = copy(scalar_factors)

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
        if isempty(tau.neighbours)
            # all variables of connected component are eliminated
            push!(scalar_factors, tau)
        end

        # variable successfully eliminated
        delete!(factor_nodes, node)
    end

    # result is unnormalised factor over remaining variables, ∑ exp(factor) = evidence
    return get_variable_elimination_result(factor_nodes, scalar_factors)
end

export variable_elimination