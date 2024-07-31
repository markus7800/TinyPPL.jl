


# support is calculated by iterating over the product of all parent supports
# thus, parents must already have support
function get_initial_aqua_support(pgm::PGM, node::VariableNode, parents::Vector{VariableNode}, N::Int, l_prec::Float64, r_prec::Float64)
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    is_observed = isobserved(pgm, node.variable)
    if is_observed
        # observed value is static
        return Float64[get_observed_value(pgm, node.variable)]
    end

    # we assume that distributions are either discrete or continuous we do not check however
    dist_type = nothing
    is_bounded = false

    x0 = Inf
    x1 = -Inf

    # iterate over the product of all parent supports
    for assignment in Iterators.product([parent.support for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, value) in zip(parents, assignment)
            X[parent.variable] = value
        end
        d = dist(X)
        if isnothing(dist_type)
            dist_type = typeof(d)
        else
            @assert typeof(d) == dist_type
        end
        if d isa Distributions.ContinuousUnivariateDistribution
            if Distributions.isbounded(d)
                x0 = min(x0, minimum(d))
                x1 = max(x1, maximum(d))
                is_bounded = true
            else
                x0 = min(x0, Distributions.quantile(d, l_prec))
                x1 = max(x1, Distributions.quantile(d, 1-r_prec))
            end
        elseif d isa Distributions.DiscreteUnivariateDistribution
            if Distributions.hasfinitesupport(d)
                x0 = min(x0, minimum(d))
                x1 = max(x1, maximum(d))
                is_bounded = true
            else
                x0 = min(x0, Distributions.quantile(d, l_prec))
                x1 = max(x1, Distributions.quantile(d, 1-r_prec))
            end
        else
            error("Distribution must be continuous")
        end
    end
    if dist_type <: Distributions.DiscreteUnivariateDistribution
        return Vector{Float64}(x0:x1), is_bounded # stepsize == 1 always
    else
        return Vector{Float64}(LinRange(x0, x1, N)), is_bounded
    end
end

# conditional probability distribution p(x|parents(x)) for unobserved
# likelihood p(y=e|parents(y)) for observed
function get_aqua_table(pgm::PGM, node::VariableNode, parents::Vector{VariableNode}, logscale::Bool)
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    is_observed = isobserved(pgm, node.variable)
    Δ = is_observed ? 0. : (length(node.support) == 1 ? 1. : (node.support[2] - node.support[1]))

    if !is_observed
        cpd = zeros([length(parent.support) for parent in parents]..., length(node.support))
    else
        cpd = zeros([length(parent.support) for parent in parents]...)
    end

    for assignment in Iterators.product([1:length(parent.support) for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, i) in zip(parents, assignment)
            X[parent.variable] = parent.support[i]
        end
        d = dist(X)
        if !is_observed
            for (i, x) in enumerate(node.support)
                cpd[assignment..., i] = logscale ? logpdf(d, x) + log(Δ) : pdf(d, x) * Δ
            end
        else
            x = get_observed_value(pgm, node.variable)
            cpd[assignment...] = logscale ? logpdf(d, x) : pdf(d, x)
        end
    end

    return cpd
end

function get_aqua_factor_graph(pgm::PGM, N::Int,
    marginal_density_cubes::Dict{Any,Tuple{Vector{Float64},Vector{Float64}}},
    expansion_counter::Dict{Any,Int};
    logscale::Bool=true, density_thresh::Float64=1e-5)

    MAX_N_EXPANSIONS = 10
    BOUNDED_FLAG = 100

    # create a variable node for each PGM variable
    variable_nodes = [VariableNode(i, pgm.addresses[i]) for i in 1:pgm.n_variables]
    factor_nodes = FactorNode[]
    did_update_support = false
    for v in pgm.topological_order
        # get all parent variabel nodes pa(v)
        parents = VariableNode[variable_nodes[x] for (x,y) in pgm.edges if y == v]
        node = variable_nodes[v]
        if !isobserved(pgm, v)
            if !haskey(marginal_density_cubes, node.address)
                # initialisation
                node.support, is_bounded = get_initial_aqua_support(pgm, node, parents, N, 1e-3, 1e-3)
                if is_bounded
                    expansion_counter[node.address] = BOUNDED_FLAG
                else
                    expansion_counter[node.address] = 0
                end
                # println("init ", node.address, " is bounded: ", is_bounded)

                did_update_support = true
            else
                xs, ps = marginal_density_cubes[node.address]
                @assert !any(isnan.(ps))

                node.support = xs
                ec = expansion_counter[node.address]

                x0, x1 = xs[1], xs[end]
                Δ = length(xs) == 1 ? 1. : (xs[2] - xs[1])
                is_discrete = Δ ≈ 1 # hack, but should work

                should_try_shrink = true

                # first we try to find an interval such that the ends are below density_thresh
                if ec < MAX_N_EXPANSIONS
                    if ps[1] > density_thresh
                        ec += 1
                        x0 -= Δ * max(1, (length(xs) ÷ 10)) # increase interval ~10%
                    end
                    if ps[end] > density_thresh
                        ec += 1
                        x1 += Δ * max(1,(length(xs) ÷ 10))
                    end
                    if ps[1] < density_thresh && ps[end] < density_thresh
                        # we are finished with expanding interval -> set to max
                        expansion_counter[node.address] = MAX_N_EXPANSIONS
                    else
                        # node.support = get_initial_aqua_support(pgm, node, parents, N, 10.0^(-l_prec), 10.0^(-r_prec))
                        if is_discrete
                            node.support = Vector{Float64}(x0:x1)
                        else
                            node.support = Vector{Float64}(LinRange(x0, x1, N))
                        end
                        did_update_support = true
                        println(node.address)
                        println("old support: ", xs[1], " ... ", xs[end])#, " ", largest_bounds[node.address])
                        println("new support: ", node.support[1], " ... ", node.support[end])#, " ($l_prec, $r_prec)")
                        largest_bounds[node.address] = (l_prec, r_prec)

                        should_try_shrink = false
                    end
                end

                if should_try_shrink && ec != BOUNDED_FLAG
                    # now we try to shrink interval to area where density > density_thresh
                    i = 1
                    while ps[i] < density_thresh && i < length(xs) / 2
                        i += 1
                    end
                    x0 = xs[i]

                    j = length(xs)
                    while ps[j] < density_thresh && j > length(xs) / 2
                        j -= 1
                    end
                    x1 = xs[j]

                    if is_discrete
                        xs_new = Vector{Float64}(x0:x1)
                    else
                        xs_new = Vector{Float64}(LinRange(x0, x1, N))
                    end

                    
                    node.support = xs_new
                    did_update_support = did_update_support || 1 < i || j < length(xs)

                    if did_update_support
                        println(node.address)
                        println("old support: ", xs[1], " ... ", xs[end])
                        println("new support: ", xs_new[1], " ... ", xs_new[end])
                    end
                end
            end
            # create factor node that represents CPD p(v | pa(v))
            cpd = get_aqua_table(pgm, node, parents, logscale)
            # factor consists of parents ∪ {node}
            factor_node = FactorNode(push!(parents, node), cpd)
        else
            # each observation is represented by one factor  p(observed_value | pa(v))
            cpd = get_aqua_table(pgm, node, parents, logscale)
            # factor consists only of parents
            factor_node = FactorNode(parents, cpd)
        end
        # connect variable nodes to factor node
        for neighbour in factor_node.neighbours
            push!(neighbour.neighbours, factor_node)
        end
        push!(factor_nodes, factor_node)
    end
    # likelihood of observed variables is integrated in factor
    variable_nodes = variable_nodes[1:pgm.n_latents]

    return variable_nodes, factor_nodes, did_update_support
end

function aqua(pgm::PGM, N::Int; method::Symbol=:bp)
    @assert method in (:ve, :bp, :jt) # variable elimination or belief propagation  

    result = Dict{Any,Tuple{Vector{Float64},Vector{Float64}}}()
    expansion_counter = Dict{Any,Int}()

    count = 0
    while true
        count += 1
        count > 100 && break

        variable_nodes, factor_nodes, did_update_support = get_aqua_factor_graph(pgm, N, result, expansion_counter)
        !did_update_support && break
        
        if method == :ve
            for node in variable_nodes
                f, Z = variable_elimination(pgm, variable_nodes, factor_nodes, [node.variable], :Greedy)
                Δ = length(node.support) == 1 ? 1. : node.support[2] - node.support[1] # is not observed variable_node
                result[node.address] =  (node.support, exp.(f.table) / (Z * Δ))
            end
        elseif method == :bp
            
            if !is_tree(variable_nodes, factor_nodes)
                # print_dot(variable_nodes, factor_nodes)
                error("Factor graph is not a tree :(")
            end
            f, evidence, marginals = belief_propagation(factor_nodes[1], true)
            for (node, ps) in marginals
                Δ = node.support[2] - node.support[1]
                result[node.address] = (node.support, ps / Δ)
            end
        elseif method == :jt
            elimination_order = get_elimination_order(pgm, variable_nodes, Int[], :Greedy)
            junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, factor_nodes[1])
            f, evidence, marginals = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, true)
            for (node, ps) in marginals
                Δ = node.support[2] - node.support[1]
                result[node.address] = (node.support, ps / Δ)
            end
        end
    end

    return result
end
export aqua