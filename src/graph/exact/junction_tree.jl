

function get_junction_tree(pgm::PGM)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    elimination_order = get_elimination_order(pgm, variable_nodes, Int[])
    return get_junction_tree(factor_nodes, elimination_order, return_factor)
end

mutable struct ClusterNode
    cluster::Vector{VariableNode}
    neighbours::Set{ClusterNode}
    factors::Set{FactorNode}
    parent::Union{ClusterNode, Nothing}
    messages::Vector{Array{Float64}}
    function ClusterNode(cluster::Vector{VariableNode})
        return new(cluster, Set{ClusterNode}(), Set{FactorNode}(), nothing, Vector{Array{Float64}}())
    end
end
function Base.show(io::IO, cluster_node::ClusterNode)
    print(io, "ClusterNode(", [node.address for node in cluster_node.cluster], ")")
end

function get_junction_tree(factor_nodes::Vector{FactorNode}, elimination_order::Vector{VariableNode}, root_factor::FactorNode)
    # println("elimination_order: ", elimination_order)

    # simulate variable elimination_order

    factor_nodes = Set(factor_nodes)
    # println("factor_nodes: ", factor_nodes)
    # println()
    tau_to_psi = Dict()
    junction_tree = ClusterNode[]

    for node in elimination_order
        # println("node to eliminate: ", node)
        neighbour_factors = Set(f for f in factor_nodes if node in f.neighbours)
        # println("neighbour_factors: ", neighbour_factors)

        for f in neighbour_factors
            delete!(factor_nodes, f)
        end

        psi = foldl((x,y) -> x ∪ y.neighbours, neighbour_factors, init=Set{VariableNode}())
        cluster_node = ClusterNode(collect(psi))
        push!(junction_tree, cluster_node)

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
        mask = trues(length(junction_tree))
        for (i, cluster_node) in enumerate(junction_tree)
            for neighbour in cluster_node.neighbours
                if cluster_node.cluster ⊆ neighbour.cluster && length(cluster_node.cluster) < length(neighbour.cluster)
                    # neighbour takes all edges
                    for n in cluster_node.neighbours
                        if n != neighbour
                            push!(neighbour.neighbours, n)
                        end
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
                    # println("merge ", cluster_node, " in ", neighbour)
                    mask[i] = false
                    did_change = true
                    break
                end
            end
            did_change && break
        end
        junction_tree = junction_tree[mask]
    end


    root_cluster_node = junction_tree[findfirst(x -> root_factor in x.factors, junction_tree)]
    make_directed(root_cluster_node, nothing)

    return junction_tree
end

export get_junction_tree
function junction_tree_message_passing(pgm::PGM)
    junction_tree = get_junction_tree(pgm)
    junction_tree_message_passing(pgm, junction_tree)
end

function make_directed(node::ClusterNode, parent::Union{ClusterNode, Nothing})
    node.parent = parent
    for neighbour in node.neighbours
        if neighbour == parent
            continue
        end
        make_directed(neighbour, node)
    end
end

function junction_tree_message_passing(pgm::PGM, junction_tree::Vector{ClusterNode})
    root = junction_tree[findfirst(x -> isnothing(x.parent), junction_tree)]
    println(root)
    # for cluster_node in junction_tree
    #     println(cluster_node, ": ", cluster_node.neighbours, " - parent: ", cluster_node.parent)
    # end
end

export junction_tree_message_passing