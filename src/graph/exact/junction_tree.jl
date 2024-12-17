

function get_junction_tree(pgm::PGM; order::Symbol=:Greedy, return_factor_as_root::Bool=false)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    if return_factor_as_root
        return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
        root_factor = return_factor
    else
        root_factor = factor_nodes[1]
    end
    elimination_order = get_elimination_order(pgm, variable_nodes, Int[], order)
    return get_junction_tree(variable_nodes, elimination_order, root_factor)
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
    neighbor_to_ix::Dict{ClusterNode,Int}
    messages::Vector{FactorNode} # messages to node from neigbours in order of neighbor_to_ix
    potential::FactorNode # psi_i = ∏ f for f in factors
    belief::FactorNode # β_i
    function ClusterNode(cluster::Vector{VariableNode})
        potential = FactorNode(cluster, zeros(Tuple([length(v.support) for v in cluster])))
        return new(cluster, Set{ClusterNode}(), Set{FactorNode}(), nothing, Dict{ClusterNode,Int}(), Vector{FactorNode}(), potential, potential)
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

    # TODO: check connectedness


    if maximal_clique
        # PGM 10.4.1
        # It is standard to reduct the tree to contain only clusters that are maximal cliques.
        # Specifically, we eliminate from the tree a cluster C_j which is a strict subset of some other cluster.
        # removes one edge at a time, could be improved, but is fast enough even for large networks
        did_change = true
        while did_change
            did_change = false
            ix = 0
            for (i, cluster_node) in enumerate(junction_tree)
                for neighbour in cluster_node.neighbours
                    if cluster_node.cluster ⊆ neighbour.cluster && length(cluster_node.cluster) < length(neighbour.cluster)
                        merge_cluster_node_into_neighbour(cluster_node, neighbour)
                        # println("merge ", cluster_node, " in ", neighbour)
                        ix = i
                        did_change = true
                        break
                    end
                end
                did_change && break
            end
            if did_change
                deleteat!(junction_tree,ix)
            end
        end
    end

    
    for clusternode in junction_tree
        for (ix, neighbour) in enumerate(clusternode.neighbours)
            clusternode.neighbor_to_ix[neighbour] = ix
        end
    end

    # PGM 10.1.2 Theorem 10.1
    # The cluster graph induced by an execution of variable elimination is necessarily a tree.
    # If whenever there is a variable x such that x ∈ C_i and x ∈ C_j then x is also in every cluster
    # on the unique path from C_i to C_j. In particular, there is a path.
    # This (running intersection) property makes the cluster tree a junction tree.
    if !isempty(root_factor.neighbours)
        root_cluster_node = junction_tree[findfirst(x -> root_factor in x.factors, junction_tree)]
    else
        root_cluster_node = junction_tree[1]
    end
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


function print_dot_junction_tree(junction_tree::Vector{ClusterNode})
    println("graph {")
    println("  node[shape=box]")
    for clusternode in junction_tree
        clusternode_str = replace(string(clusternode), "\""=>"\\\"")
        for node in clusternode.neighbours
            if node != clusternode.parent
                node_str = replace(string(node), "\""=>"\\\"")
                println("  \"$clusternode_str\" -- \"$node_str\"")
            end
        end
    end
    println("}")
end
export print_dot_junction_tree

function junction_tree_message_passing(pgm::PGM; calibrate_tree::Bool=false, return_factor_as_root::Bool=false)
    junction_tree, root_cluster_node, root_factor = get_junction_tree(pgm, return_factor_as_root=return_factor_as_root)
    junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, calibrate_tree)
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
function forward(node::ClusterNode, parent::Union{ClusterNode, Nothing})::FactorNode
    # (10.2) δ_{i → j} = ∑_{C_i - S_ij} ψ_i ∏_{k ∈ ne_i - j} δ_{k → j}
    # is a factor with scope S_ij where S_ij = C_i ∩ C_j for clique tree
    # i = node
    # j = parent
    message = node.potential # ψ_i
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == parent && continue # k ∈ ne_i - j
        child_message = forward(neighbour, node) #  δ_{k → j}
        node.messages[i] = child_message
        message = factor_product(message, child_message)
    end

    if !isnothing(parent)
        message = factor_sum(message, setdiff(node.cluster, parent.cluster))
    else
        # only at root, computes evidence
        message = factor_sum(message, node.cluster)
    end

    return message
end

function backward(node::ClusterNode, parent::Union{ClusterNode, Nothing})
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == parent && continue
        
        # sum over all messages except neighbour (a bit wastefull see backward_with_division)
        child_message = node.potential
        for (j, other_neighbor) in enumerate(node.neighbours)
            other_neighbor == neighbour && continue
            child_message = factor_product(child_message, node.messages[j])
        end

        index_in_child = neighbour.neighbor_to_ix[node]
        neighbour.messages[index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))

        backward(neighbour, node)
    end
end

function backward_with_division(node::ClusterNode, parent::Union{ClusterNode, Nothing})
    base_message = reduce(factor_product, node.messages, init=node.potential)
    for (i, neighbour) in enumerate(node.neighbours)
        neighbour == parent && continue
        
        # child_message = node.potential
        # for (j, n) in enumerate(node.neighbours)
        #     n == neighbour && continue
        #     child_message = factor_product(child_message, node.messages[j])
        # end

        # PGM 10.3.1: Message Passing with Division
        div_vars = reduce(∪,
            Set(node.messages[j].neighbours) for (j,n) in enumerate(node.neighbours) if n !== neighbour;
            init=Set{VariableNode}(node.potential.neighbours)
        )
        div_vars = sort!(collect(div_vars))
        # @assert div_vars == child_message.neighbours (div_vars, child_message.neighbours)
        _child_message = EmptyFactorNode(div_vars)
        # is it okay to have 0 / 0 = 0 ? 
        factor_division!(base_message, node.messages[i], _child_message)

        # f1 = isfinite.(_child_message.table)
        # f2 = isfinite.(child_message.table)
        # @assert all(f1 .<= f2) # _child_message may have more -inf
        # @assert _child_message.table[f1] ≈ child_message.table[f1] # should agree on rest
        # we have not _child_message.table ≈ child_message.table

        child_message = _child_message

        index_in_child = neighbour.neighbor_to_ix[node]
        neighbour.messages[index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))

        backward_with_division(neighbour, parent)
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

struct JunctionTreeMessagePassingResult
    is_calibrated::Bool
    junction_tree::Vector{ClusterNode}
    root::ClusterNode
    root_factor::FactorNode
    evidence::Float64
end
Base.show(io::IO, tree::JunctionTreeMessagePassingResult) = print(io, "JunctionTreeMessagePassingResult()")

function calibrate_bliefs(junction_tree::Vector{ClusterNode})
    # compute β_i(C_i) = ψ_i ∏ δ_{k → i} = ∑_{X - C_i} P(X)
    for node in junction_tree
        node.belief = reduce(factor_product, node.messages, init=node.potential)
    end
end

function junction_tree_message_passing(junction_tree::Vector{ClusterNode}, root::ClusterNode, root_factor::FactorNode, calibrate_tree::Bool; with_division::Bool=false)
    @assert root_factor in root.factors || isempty(root_factor.neighbours)
    initialise_potentials(root)

    res = forward(root, nothing)
    evidence = exp(res.table[1])

    if calibrate_tree
        # only need backward pass if we want to evaluate all marginals
        if with_division
            backward_with_division(root, nothing)
        else
            backward(root, nothing)
        end
        calibrate_bliefs(junction_tree)
    end

    return JunctionTreeMessagePassingResult(calibrate_tree, junction_tree, root, root_factor, evidence)
end

export junction_tree_message_passing

function get_posterior_for_root_factor(res::JunctionTreeMessagePassingResult)
    # The message δ_{k → i}(S_ki) multiplies all factors that are reachable from i through k
    # Thus, β_i(C_i) = ψ_i ∏ δ_{k → i} = ∑_{X - C_i} P(X) (Corollary 10.2)
    # This, holds for root C_r after forward pass and for all other nodes after backward pass (Corollary 10.1)
    root = res.root
    root_factor = res.root_factor
    return_factor = reduce(factor_product, root.messages, init=root.potential)
    return_factor = factor_sum(return_factor, setdiff(return_factor.neighbours, root_factor.neighbours))
    return exp.(return_factor.table) ./ res.evidence
end
export get_posterior_for_root_factor

function get_marginals(res::JunctionTreeMessagePassingResult)
    @assert res.is_calibrated

    variable_nodes = get_variable_nodes(res.junction_tree)
       
    marginals = Vector{Tuple{VariableNode, Vector{Float64}}}(undef, length(variable_nodes))
    cached_factors = Dict{ClusterNode, FactorNode}()
    for (i, (v,cluster_node)) in enumerate(variable_nodes)
        # cluster is the smallest cluster that v belongs to
        # if we have not already done for other variable that belongs to cluster, cache the result
        # sum out all other variables to get marginal of v
        factor = factor_sum(cluster_node.belief, setdiff(cluster_node.belief.neighbours, [v]))

        table = exp.(factor.table)
        table /= sum(table)
        marginals[i] = (v, table)
    end

    return marginals
end
export get_marginals



# PGM Algorithm 10.4
function query(res::JunctionTreeMessagePassingResult, marginal_variables::Vector{Int})
    @assert res.is_calibrated

    # collect all clusters which contain at least on marginal variable
    query_nodes = Set{ClusterNode}()
    for node in res.junction_tree
        if !isempty(marginal_variables ∩ map(v -> v.variable, node.cluster))
            push!(query_nodes, node)
        end
    end
    
    # compute subtree that contains all nodes
    path_to_roots = Vector{Vector{ClusterNode}}()
    for node in query_nodes
        path_to_root = ClusterNode[node]
        while !isnothing(node.parent)
            node = node.parent
            pushfirst!(path_to_root, node)
        end
        push!(path_to_roots, path_to_root)
    end
    @assert allequal(path[1] for path in path_to_roots)
    root = path_to_roots[1][1]
    i = 1
    while all(length(path) >= i for path in path_to_roots) && allequal(path[i] for path in path_to_roots)
        root = path_to_roots[1][i]
        i += 1
    end

    subtree = copy(query_nodes)
    push!(subtree, root)
    for path in path_to_roots
        for node in path[i:end]
            push!(subtree, node)
        end
    end

    # println(root)
    # println(subtree)

    # verify root
    for node in subtree
        _root = node
        while !isnothing(_root.parent) && _root.parent in subtree
            _root = _root.parent
        end
        @assert _root == root
    end


    # compute factors
    factor_nodes = FactorNode[]
    for node in subtree
        if node == root
            factor_node = node.belief
        else
            factor_node = node.belief
            message_from_parent = node.parent.messages[node.parent.neighbor_to_ix[node]]
            message_to_parent = node.messages[node.neighbor_to_ix[node.parent]]
            μ = factor_product(message_from_parent, message_to_parent)
            factor_node = factor_division!(factor_node, μ, EmptyFactorNode(factor_node.neighbours))
        end
        # copy here in order to not modify node.belief neighbours
        push!(factor_nodes, FactorNode(copy(factor_node.neighbours), factor_node.table))
    end
    
    # construct new variables for subtree (they are neighbours only of the computed factors)
    variable_nodes_dict = Dict{Int,VariableNode}()
    for factor_node in factor_nodes
        for (i,v) in enumerate(factor_node.neighbours)
            if !haskey(variable_nodes_dict, v.variable)
                new_variable_node = VariableNode(v.variable, v.address)
                new_variable_node.support = copy(v.support)
                variable_nodes_dict[v.variable] = new_variable_node
            end
            variable_node = variable_nodes_dict[v.variable]
            factor_node.neighbours[i] = variable_node
            push!(variable_node.neighbours, factor_node)
        end
    end
    variable_nodes = collect(values(variable_nodes_dict))
    
    # perform variable elimination
    elimination_order = get_greedy_elimination_order(variable_nodes, marginal_variables)
    return variable_elimination(variable_nodes, elimination_order)
end
export query

query(res::JunctionTreeMessagePassingResult, marginal_variables::Vector{VariableNode}) = query(res, Int[v.variable for v in marginal_variables])


function sample_clusternode(res::JunctionTreeMessagePassingResult, node::ClusterNode)
    @assert res.is_calibrated


    ps = exp.(node.belief.table) ./ res.evidence
    @assert sum(ps) ≈ 1

    c = CartesianIndices(ps)[rand(Categorical(reshape(ps,:)))]
    println(c, ", ", ps[c], ", ", log(ps[c]))

    I = similar(node.belief)
    I.table .= -Inf
    I.table[c] = 0

    X = Dict{Int,Int}()
    for (i,v) in enumerate(node.belief.neighbours)
        X[v.variable] = c[i]
    end

    for neighbour in node.neighbours
        child_message = factor_product(I, node.potential)
        for (j, n) in enumerate(node.neighbours)
            n == neighbour && continue
            child_message = factor_product(child_message, node.messages[j])
        end

        index_in_child = neighbour.neighbor_to_ix[node]
        neighbour.messages[index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))


        new_message = factor_product(I, node.belief)
        new_message = factor_sum(new_message, setdiff(node.cluster, neighbour.cluster))

        index_in_child = neighbour.neighbor_to_ix[node]
        old_message = neighbour.messages[index_in_child]

        println(new_message.table)
        println(old_message.table)

        new_neighbour_messages = copy(neighbour.messages)
        new_neighbour_messages[index_in_child] = new_message

        new_neighbour_belief = reduce(factor_product, new_neighbour_messages, init=neighbour.potential)
        new_neighbour_belief_table = exp.(new_neighbour_belief.table)
        println(sum(new_neighbour_belief_table), " vs ", sum(exp,neighbour.belief.table))
        # display(new_neighbour_belief_table ./ sum(new_neighbour_belief_table))


        table_sel = [get(X, v.variable, Colon()) for v in new_neighbour_belief.neighbours]
        new_belief_table_selected = new_neighbour_belief_table[table_sel...]
        println(sum(new_belief_table_selected))
        println(size(new_belief_table_selected))

        old_belief_table_selected = exp.(neighbour.belief.table[table_sel...])
        println(sum(old_belief_table_selected), " vs ", ps[c] * res.evidence)
        println(size(old_belief_table_selected))

        println("diff: ", maximum(abs,
            (new_belief_table_selected ./ sum(new_belief_table_selected)) .-
            (old_belief_table_selected ./ sum(old_belief_table_selected))
        ))
        println((new_belief_table_selected ./ sum(new_belief_table_selected)) ≈ (old_belief_table_selected ./ sum(old_belief_table_selected)))
    end

end

export sample_clusternode

function dfs_order(node::ClusterNode, parent::Union{Nothing,ClusterNode}, nodes::Vector{ClusterNode})
    push!(nodes, node)
    for neighbour in node.neighbours
        neighbour == parent && continue
        dfs_order(neighbour, node, nodes)
    end
    return nodes
end

function sample_junctiontree_naive(res::JunctionTreeMessagePassingResult)
    @assert res.is_calibrated
    messages = Dict{ClusterNode, Vector{FactorNode}}(node => copy(node.messages) for node in res.junction_tree)
    X = Dict{Int,Int}()
    sampled = Dict{ClusterNode,Bool}()
    P = Float64[res.evidence]
    for (i,node) in enumerate(dfs_order(res.root, nothing, ClusterNode[]))# res.junction_tree
        print("$i/$(length(res.junction_tree)). ")
        _sample_junctiontree_naive(res, sampled, P, messages, X, node, nothing)
    end
    return X, prod(P), messages
end
export sample_junctiontree_naive

function _sample_junctiontree_naive(res::JunctionTreeMessagePassingResult, sampled::Dict{ClusterNode,Bool}, P::Vector{Float64}, messages::Dict{ClusterNode,Vector{FactorNode}}, X::Dict{Int,Int}, node::ClusterNode, parent::Union{Nothing,ClusterNode})

    if isnothing(parent)
        sampled[node] = true
        belief = reduce(factor_product, messages[node]; init=node.potential)
        # println("\nReceived Messages at ", node)
        # for (i,neighbour) in enumerate(node.neighbours)
        #     println(neighbour, ": ")
        #     println(messages[node][i].table)
        # end

        ps = exp.(belief.table)
        Z = sum(ps)
        ps = ps / Z

        c = CartesianIndices(ps)[rand(Categorical(reshape(ps,:)))]
        println("Sample ", node, ": c=", c, ", ps[c]=", ps[c], ", Z=", Z, ", P=",prod(P))
        @assert Z ≈ prod(P)

        push!(P, ps[c])

        # add factor I to cluster node potential
        I = similar(belief)
        I.table .= -Inf
        I.table[c] = 0

        for (i,v) in enumerate(belief.neighbours)
            X[v.variable] = c[i]
        end

        node_potential = factor_product(I, node.potential)

    elseif get(sampled, node, false)

        # factor I was already added to cluster node potential
        I = similar(node.potential)
        I.table .= -Inf
        I.table[[X[v.variable] for v in node.potential.neighbours]...] = 0

        node_potential = factor_product(I, node.potential)

    else
        node_potential = node.potential
    end

    # backward
    for neighbour in node.neighbours
        neighbour == parent && continue
        child_message = node_potential
        for (j, other_neighbor) in enumerate(node.neighbours)
            other_neighbor == neighbour && continue
            child_message = factor_product(child_message, messages[node][j])
        end

        index_in_child = neighbour.neighbor_to_ix[node]
        messages[neighbour][index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))
        # println("Send message from ", node, " to ", neighbour, " at index ", index_in_child)
        # println(messages[neighbour][index_in_child].table)
        _sample_junctiontree_naive(res, sampled, P, messages, X, neighbour, node)
    end

end


function sample_junctiontree_naive_2(res::JunctionTreeMessagePassingResult)
    @assert res.is_calibrated
    messages = Dict{ClusterNode, Vector{FactorNode}}(node => copy(node.messages) for node in res.junction_tree)
    X = Dict{Int,Int}()
    sampled = Dict{ClusterNode,Bool}()
    P = Float64[res.evidence]
    _sample_junctiontree_naive_2(res, sampled, P, res.evidence, messages, X, res.root, nothing)
    return X, prod(P), messages
end
export sample_junctiontree_naive_2

function _sample_junctiontree_naive_2(res::JunctionTreeMessagePassingResult, sampled::Dict{ClusterNode,Bool}, P::Vector{Float64}, P_current::Float64, messages::Dict{ClusterNode,Vector{FactorNode}}, X::Dict{Int,Int}, node::ClusterNode, parent::Union{Nothing,ClusterNode})
    @assert !get(sampled, node, false)
    sampled[node] = true
    belief = reduce(factor_product, messages[node]; init=node.potential)

    # println("\nReceived Messages at ", node)
    # for (i,neighbour) in enumerate(node.neighbours)
    #     println(neighbour, ": ")
    #     println(messages[node][i].table)
    # end

    ps = exp.(belief.table)
    Z = sum(ps)
    ps = ps / Z
    @assert P_current ≈ Z # we only have messages (updated evidence) from nodes above node (on path from node to root)

    table_sel = [get(X, v.variable, Colon()) for v in node.belief.neighbours]
    ps2 = exp.(node.belief.table[table_sel...])
    Z2 = sum(ps2)
    ps2 = ps2 / Z2
    @assert sum(ps[table_sel...]) ≈ 1.
    @assert ps[table_sel...] ≈ ps2
    


    c = CartesianIndices(ps)[rand(Categorical(reshape(ps,:)))]
    println("Sample ", node, ": c=", c, ", ps[c]=", ps[c], ", Z=", Z, ", Z2=", Z2, ", P=", P_current)
    push!(P, ps[c])
    P_current *= ps[c]

    I = similar(belief)
    I.table .= -Inf
    I.table[c] = 0

    for (i,v) in enumerate(belief.neighbours)
        X[v.variable] = c[i]
    end

    node_potential = factor_product(I, node.potential)
    

    # backward
    for neighbour in node.neighbours
        neighbour == parent && continue
        child_message = node_potential
        for (j, other_neighbor) in enumerate(node.neighbours)
            other_neighbor == neighbour && continue
            child_message = factor_product(child_message, messages[node][j])
        end

        index_in_child = neighbour.neighbor_to_ix[node]
        messages[neighbour][index_in_child] = factor_sum(child_message, setdiff(node.cluster, neighbour.cluster))
        # println("Send message from ", node, " to ", neighbour, " at index ", index_in_child)
        # println(messages[neighbour][index_in_child].table)

        _sample_junctiontree_naive_2(res, sampled, P, P_current, messages, X, neighbour, node)
    end

    # forward
    # if !isnothing(node.parent)
    #     parent_message = factor_product(I, node.potential)
    #     for (i, neighbour) in enumerate(node.neighbours)
    #         neighbour == node.parent && continue
    #         parent_message = factor_product(parent_message, messages[node][i])
    #     end
    #     parent_message = factor_sum(parent_message, setdiff(node.cluster, node.parent.cluster))
    #     messages[node.parent][node.parent.neighbor_to_ix[node]] = parent_message
    # end
end