

function get_junction_tree(pgm::PGM; order::Symbol=:Greedy)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    elimination_order = get_elimination_order(pgm, variable_nodes, Int[], order)
    return get_junction_tree(variable_nodes, elimination_order, return_factor)
end

# PGM 10.1.1
# a cluster graph for a set of factors is a an undirected graph,
# each of whose nodes i is associated  with a subset C_i ⊆ X.
# Each factor f is associated with a cluster i such that Scope[f] ⊆ C_i
# Because each factor is assinged to exactly one clique we have that
# P = ∏ f = ∏ psi_i
mutable struct ClusterNode
    cluster::Vector{VariableNode} # C_i
    neighbours::Set{ClusterNode}
    factors::Set{FactorNode} # each factor f that is associated with ClusterNode, Scope[f] ⊆ C_i
    parent::Union{ClusterNode, Nothing}
    messages::Vector{FactorNode}
    potential::FactorNode # psi_i = ∏ f for f in factors
    function ClusterNode(cluster::Vector{VariableNode})
        potential = FactorNode(cluster, zeros(Tuple([length(v.support) for v in cluster])))
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

function merge_cluster_node_into_neighbour(cluster_node::ClusterNode, neighbour::ClusterNode)
    for n in cluster_node.neighbours
        if n != neighbour
            # neighbour takes all edges
            push!(neighbour.neighbours, n)
            push!(n.neighbours, neighbour)
        end
        # remove cluster_node from tree
        delete!(n.neighbours, cluster_node)
    end
    for f in cluster_node.factors
        push!(neighbour.factors, f)
    end
end

# PGM: 10.4
function get_junction_tree(variable_nodes::Vector{VariableNode}, elimination_order::Vector{VariableNode}, root_factor::FactorNode, maximal_clique::Bool=true)

    # The execution of a variable elimination algorithm can be associated with a cluster graph.
    # A cluster C_i corresponds to the factor psi_i generated during the execution of the algorithm.
    # And an undirected edge connects C_i and C_j when tau_i is used (directly) in the computation of psi_j or vice-versa.

    factor_nodes = Dict(v => Set(v.neighbours) for v in variable_nodes)

    tau_to_cluster_node = Dict{FactorNode,ClusterNode}()
    junction_tree = ClusterNode[]

    for node in elimination_order
        neighbour_factors = factor_nodes[node]
        # println("node to eliminate: ", node)
        # println("neighbour_factors: ", neighbour_factors)

        # mock variable elimination
        # create psi
        #psi = foldl((x,y) -> x ∪ y.neighbours, neighbour_factors, init=Set{VariableNode}())
        psi = reduce(∪, Set(f.neighbours) for f in neighbour_factors; init=Set{VariableNode}())
        cluster_node = ClusterNode(sort!(collect(psi)))
        push!(junction_tree, cluster_node)

        for f in neighbour_factors
            if haskey(tau_to_cluster_node, f)
                # f is actually a tau_i created by eliminating variable before
                neighbour = tau_to_cluster_node[f] # get C_i
                # connect both clusters C_i (=neighbour) and C_j (=clusternode)
                push!(cluster_node.neighbours, neighbour)
                push!(neighbour.neighbours, cluster_node)
            else
                # distribute initial factors (not tau to cluster_nodes)
                # each initial factor belongs to exactly one cluster node (we delete them from factor_nodes)
                # scope(f) ⊆ cluster_node.cluster = scope(psi)
                push!(cluster_node.factors, f)
            end
        end

        # eliminate node, create tau
        delete!(psi, node)
        tau = FactorNode(sort!(collect(psi)), Float64[]) # dummy factor
        tau_to_cluster_node[tau] = cluster_node

        for f in neighbour_factors
            for v in f.neighbours
                delete!(factor_nodes[v], f)
            end
        end

        # println("tau: ", tau)
        for v in tau.neighbours
            push!(factor_nodes[v], tau)
        end

        # variable successfully eliminated
        delete!(factor_nodes, node)
    end

    if maximal_clique
        # PGM 10.4.1
        # It is standard to reduct the tree to contain only clusters that are maximal cliques.
        # Specifically, we eliminate from the tree a cluster C_j which is a strict subset of some other cluster.
        # removes one edge at a time, could be improved
        did_change = true
        while did_change
            did_change = false
            mask = trues(length(junction_tree))
            for (i, cluster_node) in enumerate(junction_tree)
                for neighbour in cluster_node.neighbours
                    if cluster_node.cluster ⊆ neighbour.cluster && length(cluster_node.cluster) < length(neighbour.cluster)
                        merge_cluster_node_into_neighbour(cluster_node, neighbour)
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
    end


    # PGM 10.1.2 Theorem 10.1
    # The cluster graph induced by an execution of variable elimination is necessarily a tree.
    # If whenever there is a variable x such that x ∈ C_i and x ∈ C_j then x is also in every cluster
    # on the unique path from C_i to C_j. In particular, there is a path.
    # This (running intersection) property makes the cluster tree a junction tree.
    root_cluster_node = junction_tree[findfirst(x -> root_factor in x.factors, junction_tree)]
    make_directed(root_cluster_node, nothing)

    return junction_tree, root_cluster_node, root_factor
end

export get_junction_tree

function print_junction_tree(root::ClusterNode, tab="")
    println(tab, root)
    for child in root.neighbours
        if child != root.parent
            print_junction_tree(child, tab*"  ")
        end
    end
end
export print_junction_tree

function junction_tree_message_passing(pgm::PGM; all_marginals::Bool=false)
    junction_tree, root_cluster_node, root_factor = get_junction_tree(pgm)
    junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, all_marginals)
end


function initialise_potentials(node::ClusterNode)
    node.potential = reduce(factor_product, node.factors, init=node.potential)
    node.messages = Vector{FactorNode}(undef, length(node.neighbours))
    for neighbour in node.neighbours
        neighbour == node.parent && continue
        initialise_potentials(neighbour)
    end
end

# PGM 10.2.1: Variable Elimination in a Clique Tree
# pass messages towards root
function forward(node::ClusterNode)::FactorNode
    # (10.2) δ_{i → j} = ∑_{C_i - S_ij} ψ_i ∏_{k ∈ ne_i - j} δ_{k → j}
    # is a factor with scope S_ij where S_ij = C_i ∩ C_j for clique tree
    # i = node
    # j = node.parent
    message = node.potential # ψ_i
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == node.parent && continue # k ∈ ne_i - j
        child_message = forward(neighbour) #  δ_{k → j}
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
    # message = reduce(factor_product, node.messages, init=node.potential)
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == node.parent && continue
        
        child_message = node.potential
        for (j, n) in enumerate(node.neighbours)
            n == neighbour && continue
            child_message = factor_product(child_message, node.messages[j])
        end

        # PGM 10.3.1: Message Passing with Division
        # child_message = factor_division(message, node.messages[i])

        index_in_child = findfirst(n -> n==node, collect(neighbour.neighbours))
        neighbour.messages[index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))

        backward(neighbour)
    end
end

# assigns variable the smallest cluster it belongs to
function get_variable_nodes(junction_tree::Vector{ClusterNode})
    variable_nodes = Dict{VariableNode, ClusterNode}()
    for cluster_node in junction_tree
        for variable in cluster_node.cluster
            if !haskey(variable_nodes, variable)
                variable_nodes[variable] = cluster_node
            else
                other_cluster_node = variable_nodes[variable]
                if length(cluster_node.cluster) < length(other_cluster_node.cluster)
                    variable_nodes[variable] = cluster_node
                end
            end
        end
    end
    return variable_nodes
end

function junction_tree_message_passing(junction_tree::Vector{ClusterNode}, root::ClusterNode, root_factor::FactorNode, all_marginals::Bool)
    @assert root_factor in root.factors
    initialise_potentials(root)

    res = forward(root)
    evidence = exp(res.table[1])

    # The message δ_{k → i}(S_ki) multiplies all factors that are reachable from i through k
    # Thus, β_i(C_i) = ψ_i ∏ δ_{k → i} = ∑_{X - C_i} P(X) (Corollary 10.2)
    # This, holds for root C_r after forward pass and for all other nodes after backward pass (Corollary 10.1)
    return_factor = reduce(factor_product, root.messages, init=root.potential)
    return_factor = factor_sum(return_factor, setdiff(return_factor.neighbours, root_factor.neighbours))

    if all_marginals
        # only need backward pass if we want to evaluate all marginals
        backward(root)

        variable_nodes = get_variable_nodes(junction_tree)
       
        marginals = Vector{Tuple{Int, Any, Vector{Float64}}}(undef, length(variable_nodes))
        cached_factors = Dict{ClusterNode, FactorNode}()
        for (i, (v,cluster_node)) in enumerate(variable_nodes)
            # cluster is the smallest cluster that v belongs to
            # compute β_i(C_i) = ψ_i ∏ δ_{k → i} = ∑_{X - C_i} P(X)
            # if we have not already done for other variable that belongs to cluster, cache the result
            factor = get!(
                cached_factors, cluster_node,
                reduce(factor_product, cluster_node.messages, init=cluster_node.potential
                )
            )
            # sum out all other variables to get marginal of v
            factor = factor_sum(factor, setdiff(factor.neighbours, [v]))

            table = exp.(factor.table)
            table /= sum(table)
            marginals[i] = (v.variable, v.address, table)
        end

        return return_factor, evidence, marginals
    else

        return return_factor, evidence
    end
end

export junction_tree_message_passing