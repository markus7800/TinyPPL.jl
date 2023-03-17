

function get_junction_tree(pgm::PGM, marginal_variables=nothing)
    marginal_variables = parse_marginal_variables(pgm, marginal_variables)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    elimination_order = get_elimination_order(pgm, variable_nodes, marginal_variables)
    return get_junction_tree(pgm, variable_nodes, factor_nodes, elimination_order)
end

mutable struct ClusterNode
    cluster::Set{VariableNode}
    neighbours::Set{ClusterNode}
    factors::Set{FactorNode}
    function ClusterNode(cluster::Set{VariableNode})
        return new(cluster, Set{ClusterNode}(), Set{FactorNode}())
    end
end
function Base.show(io::IO, cluster_node::ClusterNode)
    print(io, "ClusterNode(", [node.address for node in cluster_node.cluster], ")")
end

function get_junction_tree(pgm::PGM, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode}, elimination_order::Vector{VariableNode})
    # println("elimination_order: ", elimination_order)

    # simulate variable elimination_order

    factor_nodes = Set(factor_nodes)
    # println("factor_nodes: ", factor_nodes)
    # println()
    tau_to_psi = Dict()
    cluster_nodes = ClusterNode[]

    for node in elimination_order
        # println("node to eliminate: ", node)
        neighbour_factors = Set(f for f in factor_nodes if node in f.neighbours)
        # println("neighbour_factors: ", neighbour_factors)

        for f in neighbour_factors
            delete!(factor_nodes, f)
        end

        psi = foldl((x,y) -> x ∪ y.neighbours, neighbour_factors, init=Set{VariableNode}())
        cluster_node = ClusterNode(psi)
        push!(cluster_nodes, cluster_node)

        for f in neighbour_factors
            if haskey(tau_to_psi, f)
                neighbour = tau_to_psi[f]
                push!(cluster_node.neighbours, neighbour)
                push!(neighbour.neighbours, cluster_node)
            else
                push!(cluster_node.factors, f)
            end
        end

        psi = copy(psi)
        delete!(psi, node)
        tau = FactorNode(collect(psi), Float64[]) # dummy factor
        tau_to_psi[tau] = cluster_node
        # println("tau: ", tau)
        push!(factor_nodes, tau)

        # println("factor_nodes: ", factor_nodes)
        # println()
    end

    # removes one edge at a time, could be improved
    did_change = true
    while did_change
        did_change = false
        mask = trues(length(cluster_nodes))
        for (i, cluster_node) in enumerate(cluster_nodes)
            for neighbour in cluster_node.neighbours
                if cluster_node.cluster ⊆ neighbour.cluster && length(cluster_node.cluster) < length(neighbour.cluster)
                    # neighbour takes all edges
                    for n in cluster_node.neighbours
                        push!(neighbour.neighbours, n)
                    end
                    for f in cluster_node.factors
                        push!(neighbour.factors, f)
                    end
                        
                    # remove cluster_node from tree
                    for n in cluster_node.neighbours
                        if n != neighbour
                            push!(n.neighbours, neighbour)
                        end
                        delete!(n.neighbours, cluster_node)
                    end
                    println("merge ", cluster_node, " in ", neighbour)
                    mask[i] = false
                    did_change = true
                    break
                end
            end
            did_change && break
        end
        cluster_nodes = cluster_nodes[mask]
    end

    return cluster_nodes
end

export get_junction_tree