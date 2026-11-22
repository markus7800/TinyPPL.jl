


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
            @assert !(d isa Dirac)
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
        return midpoint_grid(x0, x1, N), is_bounded # parents also need to be bounded
    end
end

# N grid points that are the midpoints of N cells of equal width covering [x0, x1].
# For bounded distributions, the grid does not contain the bounds, where the density may be infinite (e.g. Beta(0.5, 0.5)).
function midpoint_grid(x0::Real, x1::Real, N::Int)
    Δ = (x1 - x0) / N
    return Vector{Float64}(LinRange(x0 + Δ/2, x1 - Δ/2, N))
end

# width of the cells represented by the grid points (1 for discrete distributions and for support with only one point)
aqua_cell_width(support::AbstractVector) = length(support) == 1 ? 1. : support[2] - support[1]

# lower and upper bound of the support of p(x | parents) over all parent values and whether the distribution is discrete
function get_aqua_support_bounds(pgm::PGM, node::VariableNode, parents::Vector{VariableNode})
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    lo, hi = Inf, -Inf
    is_discrete = false
    for assignment in Iterators.product([parent.support for parent in parents]...)
        for (parent, value) in zip(parents, assignment)
            X[parent.variable] = value
        end
        d = dist(X)
        lo = min(lo, minimum(d))
        hi = max(hi, maximum(d))
        is_discrete = d isa Distributions.DiscreteUnivariateDistribution
    end
    return lo, hi, is_discrete
end

# conditional probability distribution p(x|parents(x)) for unobserved
# likelihood p(y=e|parents(y)) for observed
function get_aqua_table(pgm::PGM, node::VariableNode, parents::Vector{VariableNode}, logscale::Bool)
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    is_observed = isobserved(pgm, node.variable)
    Δ = is_observed ? 0. : aqua_cell_width(node.support)

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

const MAX_N_EXPANSIONS = 10
const BOUNDED_FLAG = 100

function get_aqua_factor_graph(pgm::PGM, N::Int,
    marginal_density_cubes::Dict{Any,Tuple{Vector{Float64},Vector{Float64}}},
    expansion_counter::Dict{Any,Int};
    logscale::Bool=true, density_thresh::Float64=1e-5)


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
                if is_bounded && all(expansion_counter[parent.address] == BOUNDED_FLAG for parent in parents)
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
                Δ = aqua_cell_width(xs)
                lo, hi, is_discrete = get_aqua_support_bounds(pgm, node, parents)
                # continuous grid points are cell midpoints and have to stay inside the support
                margin = is_discrete ? 0. : Δ/2
                new_grid(x0, x1) = is_discrete ? Vector{Float64}(x0:x1) : Vector{Float64}(LinRange(x0, x1, N))

                should_try_shrink = true

                # first we try to find an interval such that the ends are below density_thresh
                # (or the ends are at the bounds of the support)
                if ec < MAX_N_EXPANSIONS
                    step = Δ * max(1, (length(xs) ÷ 10)) # increase interval ~10%
                    x0_new = max(x0 - step, lo + margin)
                    x1_new = min(x1 + step, hi - margin)
                    expand_left = ps[1] > density_thresh && x0_new < x0
                    expand_right = ps[end] > density_thresh && x1_new > x1
                    if !expand_left && !expand_right
                        # we are finished with expanding interval -> set to max
                        expansion_counter[node.address] = MAX_N_EXPANSIONS
                    else
                        if expand_left
                            ec += 1
                            x0 = x0_new
                        end
                        if expand_right
                            ec += 1
                            x1 = x1_new
                        end
                        node.support = new_grid(x0, x1)
                        did_update_support = true
                        @debug "expand support of $(node.address)" xs[1] xs[end] node.support[1] node.support[end]
                        expansion_counter[node.address] = ec

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

                    if 1 < i || j < length(xs)
                        node.support = new_grid(x0, x1)
                        did_update_support = true
                        @debug "shrink support of $(node.address)" xs[1] xs[end] node.support[1] node.support[end]
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

# this is more of a calibration, we adjust support for each marginal
function aqua(pgm::PGM, N::Int; method::Symbol=:bp)
    @assert method in (:ve, :bp, :jt) # variable elimination or belief propagation  

    result = Dict{Address,Tuple{Vector{Float64},Vector{Float64}}}()
    expansion_counter = Dict{Address,Int}()

    count = 0
    while true
        count += 1
        count > 100 && break

        variable_nodes, factor_nodes, did_update_support = get_aqua_factor_graph(pgm, N, result, expansion_counter)
        !did_update_support && break
        
        if method == :ve
            for node in variable_nodes
                ve_result = variable_elimination(pgm, variable_nodes, factor_nodes, [node.variable], :Greedy)
                f, log_Z = ve_result.factor, ve_result.log_evidence
                Δ = aqua_cell_width(node.support) # is not observed variable_node
                result[node.address] =  (node.support, exp.(f.table .- log_Z) / Δ)
            end
        elseif method == :bp
            
            if !is_tree(variable_nodes, factor_nodes)
                # print_dot(variable_nodes, factor_nodes)
                error("Factor graph is not a tree :(")
            end
            be_result = belief_propagation(factor_nodes[1], true)
            f, evidence, marginals = get_posterior_for_root_factor(be_result), be_result.evidence, get_marginals(be_result)
            for (node, ps) in marginals
                Δ = aqua_cell_width(node.support)
                result[node.address] = (node.support, ps / Δ)
            end
        elseif method == :jt
            elimination_order = get_elimination_order(pgm, variable_nodes, Int[], :Greedy)
            junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, factor_nodes[1]; scalar_factors=get_scalar_factors(factor_nodes))
            jt_result = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, true)
            f, evidence, marginals = get_posterior_for_root_factor(jt_result), jt_result.evidence, get_marginals(jt_result)
            for (node, ps) in marginals
                Δ = aqua_cell_width(node.support)
                result[node.address] = (node.support, ps / Δ)
            end
        end
    end

    return result
end
export aqua

function get_node_for_address(variable_nodes::Vector{VariableNode}, addr::Address)
    return first(v for v in variable_nodes if v.address == addr)
end
export get_node_for_address

function aqua_get_joint(pgm::PGM, N::Int, marginals::Dict{Address,Tuple{Vector{Float64},Vector{Float64}}}, joint::Vector{Address})

    expansion_counter = Dict{Address,Int}(addr => BOUNDED_FLAG for (addr, _) in marginals) # do not update supports anymore
    variable_nodes, factor_nodes, did_update_support = get_aqua_factor_graph(pgm, N, marginals, expansion_counter)
    @assert !did_update_support

    ve_result = variable_elimination(pgm, variable_nodes, factor_nodes, [get_node_for_address(variable_nodes, v).variable for v in joint], :Greedy)
    f, log_Z = ve_result.factor, ve_result.log_evidence
    Δ = prod(aqua_cell_width(node.support) for node in f.neighbours; init=1.)

    return [node.support for node in f.neighbours], exp.(f.table .- log_Z) / Δ
end
export aqua_get_joint

# z = 1.
# c = 0.1
# plot(t -> z <= t)
# plot!(t -> linear_kernel(t, z, c))
# plot!(t -> sigmoid_kernel(t, z, c))

# c is smoothing factor

function linear_kernel(t, z, c)
    return clamp((c + t - z) / (2*c),0,1)
end
function ∇linear_kernel(t, z, c)
    return (abs(t-z)<=c) * 1/(2*c)
end

function sigmoid_kernel(t, z, c)
    return 1/(1 + exp(-(t-z)/c))
end
function ∇sigmoid_kernel(t, z, c)
    E = exp(-abs(t-z)/c)
    return E / (c * (1 + E)^2)
end

function aqua_get_return_distribution(pgm::PGM, N::Int, marginals::Dict{Address,Tuple{Vector{Float64},Vector{Float64}}};
    mode::Symbol=:pdf, kernel::Symbol=:sigmoid, kernel_smoothing::Union{Nothing,Float64}=nothing, density_thresh::Float64=1e-5,
    z0::Union{Nothing,Float64}=nothing, z1::Union{Nothing,Float64}=nothing)
    @assert mode in (:pdf, :cdf)
    @assert kernel in (:linear, :sigmoid)

    return_variables = return_expr_variables(pgm)
    supports, joint = aqua_get_joint(pgm, N, marginals, Address[pgm.addresses[v] for v in return_variables])

    retvals = similar(joint)

    X = Vector{Float64}(undef, pgm.n_variables)
    # iterate over the product of the support of all factor variables
    for indices in Iterators.product([1:length(support) for support in supports]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (j, i) in zip(1:length(return_variables), indices)
            X[return_variables[j]] = supports[j][i]
        end
        
        retval = get_retval(pgm, X) # observed values in return expr are subsituted with their value
        @assert retval isa Real
        retvals[indices...] = retval # add return value
    end
    # transform to list
    joint = reshape(joint, :)
    retvals = reshape(retvals, :)

    z0 = isnothing(z0) ? minimum(retvals) : z0
    z1 = isnothing(z1) ? maximum(retvals) : z1
    is_discrete = all(isinteger.(retvals))

    if mode == :pdf
        # this is a trick to estimate pdf of return expression by "smoothing" with kernel

        # P(f(X,Y) ≤ z) = ∫ [f(x,y) ≤ z] p(x,y) dx dy
        #               ≈ ∫ kernel(z, f(x,y)) p(x,y) dx dy
        # p(z) = ∂/∂z P(f(X,Y) ≤ z) ≈ ∫ ∂/∂z kernel(z, f(x,y)) p(x,y) dx dy

        while true
            if is_discrete
                zs = Vector{Float64}(z0:z1)
            else
                zs = Vector{Float64}(LinRange(z0, z1, N))
            end
            println("new support: ", z0, " ... ", z1)

            kernel_smoothing = isnothing(kernel_smoothing) ? (zs[end] - zs[1]) / 200 : kernel_smoothing
            kernel_func = kernel == :linear ? ∇linear_kernel : ∇sigmoid_kernel

            Δz = zs[2] - zs[1]
            zps = zeros(length(zs))

            for i in eachindex(zps)
                z = zs[i]
                # this can be sped up by sorting retvals and only summing over neighbourhood of z
                for j in eachindex(retvals)
                    f = retvals[j]
                    p = joint[j]

                    zps[i] += kernel_func(f, z, kernel_smoothing) * p
                end
            end

            zps = zps / sum(zps) / Δz 

            i = 1
            while zps[i] < density_thresh && i < length(zs) / 2
                i += 1
            end
            z0 = zs[i]

            j = length(zs)
            while zps[j] < density_thresh && j > length(zs) / 2
                j -= 1
            end
            z1 = zs[j]

            did_update_support = 1 < i || j < length(zs)
            
            if !did_update_support
                return zs, zps
            end
        end

    else # mode == :cdf
        
        # this should be more accurate in general

        if is_discrete
            zs = Vector{Float64}(z0:z1)
        else
            zs = Vector{Float64}(LinRange(z0, z1, N))
        end
        println("new support: ", z0, " ... ", z1)

        Δz = zs[2] - zs[1]
        zps = zeros(length(zs))

        for i in eachindex(zps)
            z = zs[i]
            for j in eachindex(retvals)
                f = retvals[j]
                p = joint[j]

                zps[i] += (f <= z) * p
            end
        end

        zps = zps / zps[end]
        return zs, zps

        # i = 1
        # while zps[i] < density_thresh && i < length(zs) / 2
        #     i += 1
        # end
        # z0 = zs[i]

        # j = length(zs)
        # while zps[j] < density_thresh && j > length(zs) / 2
        #     j -= 1
        # end
        # z1 = zs[j]

        # did_update_support = 1 < i || j < length(zs)
        
        # if !did_update_support
        #     return zs, zps
        # end
    
    end
end
export aqua_get_return_distribution