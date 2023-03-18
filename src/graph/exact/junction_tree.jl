

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
    messages::Vector{FactorNode}
    potential::FactorNode
    function ClusterNode(cluster::Vector{VariableNode})
        potential = FactorNode(cluster, zeros([length(v.support) for v in cluster]...))
        return new(cluster, Set{ClusterNode}(), Set{FactorNode}(), nothing, Vector{FactorNode}(), potential)
    end
end
function Base.show(io::IO, cluster_node::ClusterNode)
    print(io, "ClusterNode(", [node.address for node in cluster_node.cluster], ")")
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

    return junction_tree, root_factor
end

export get_junction_tree
function junction_tree_message_passing(pgm::PGM, all_marginals::Bool=false)
    junction_tree, root_factor = get_junction_tree(pgm)
    junction_tree_message_passing(pgm, junction_tree, root_factor, all_marginals)
end


function initialise(node::ClusterNode)
    node.potential = reduce(factor_product, node.factors, init=node.potential)
    node.messages = Vector{FactorNode}(undef, length(node.neighbours))
    for neighbour in node.neighbours
        neighbour == node.parent && continue
        initialise(neighbour)
    end
end

function forward(node::ClusterNode)::FactorNode
    message = node.potential
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == node.parent && continue
        child_message = forward(neighbour)
        node.messages[i] = child_message
        message = factor_product(message, child_message)
    end

    if !isnothing(node.parent)
        message = factor_sum(message, setdiff(node.cluster, node.parent.cluster))
    else
        # only at root, computes evidence
        message = factor_sum(message, node.cluster)
    end

    return message
end

function backward(node::ClusterNode)
    message = reduce(factor_product, node.messages, init=node.potential)
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == node.parent && continue
        
        # message = node.potential
        # for (j, n) in enumerate(node.neighbours)
        #     n == neighbour && continue
        #     message = factor_product(message, node.messages[j])
        # end

        child_message = factor_division(message, node.messages[i])

        index_in_child = findfirst(n -> n==node, collect(neighbour.neighbours))
        neighbour.messages[index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))

        backward(neighbour)
    end
end

function junction_tree_message_passing(pgm::PGM, junction_tree::Vector{ClusterNode}, root_factor::FactorNode, all_marginals::Bool)
    root = junction_tree[findfirst(x -> isnothing(x.parent), junction_tree)]
    @assert root_factor in root.factors
    initialise(root)

    res = forward(root)
    evidence = exp(res.table[1])

    return_factor = reduce(factor_product, root.messages, init=root.potential)
    return_factor = factor_sum(return_factor, setdiff(return_factor.neighbours, root_factor.neighbours))

    if all_marginals
        backward(root)

        variable_nodes = Dict{VariableNode, ClusterNode}()
        for cluster_node in junction_tree
            for variable in cluster_node.cluster
                if !haskey(variable_nodes, variable)
                    variable_nodes[variable] = cluster_node
                else
                    other_cluster_node = variable_nodes[variable]
                    if length(other_cluster_node.cluster) > length(cluster_node.cluster)
                        variable_nodes[variable] = cluster_node
                    end
                end
            end
        end
        
        marginals = Vector{Tuple{Int, Any, Vector{Float64}}}(undef, length(variable_nodes))
        cached_factors = Dict{ClusterNode, FactorNode}()
        for (i,(v,cluster_node)) in enumerate(variable_nodes)
            factor = get!(cached_factors, cluster_node, reduce(factor_product, cluster_node.messages, init=cluster_node.potential))
            factor = factor_sum(factor, setdiff(factor.neighbours, [v]))

            table = exp.(factor.table)
            table /= sum(table)
            marginals[i] = (v.variable, v.address, table)
        end

        return return_factor, evidence, marginals
    else

        return return_factor, evidence
    end

    return evidence
end

export junction_tree_message_passing