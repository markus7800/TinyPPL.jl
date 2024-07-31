


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

    x0 = Inf
    x1 = -Inf

    # iterate over the product of all parent supports
    for assignment in Iterators.product([parent.support for parent in parents]...)
        # if parents is empty, assigment = () and next loop is skipped
        for (parent, value) in zip(parents, assignment)
            X[parent.variable] = value
        end
        d = dist(X)
        @assert d isa Distributions.ContinuousUnivariateDistribution
        # TODO: allow discrete again
        x0 = min(x0, Distributions.quantile(d, l_prec))
        x1 = max(x1, Distributions.quantile(d, 1-r_prec))
    end
    return Vector{Float64}(LinRange(x0, x1, N))
end

# conditional probability distribution p(x|parents(x)) for unobserved
# likelihood p(y=e|parents(y)) for observed
function get_aqua_table(pgm::PGM, node::VariableNode, parents::Vector{VariableNode}, logscale::Bool)
    X = Vector{Float64}(undef, pgm.n_variables)
    dist = pgm.distributions[node.variable]
    is_observed = isobserved(pgm, node.variable)
    Δ = is_observed ? 0. : (node.support[2] - node.support[1])

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
    marginal_density_cubes::Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}},
    largest_bounds::Dict{Symbol,Tuple{Int,Int}};
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
                init_prec = 3
                node.support = get_initial_aqua_support(pgm, node, parents, N, 10.0^(-init_prec), 10.0^(-init_prec))
                largest_bounds[node.address] = (init_prec, init_prec)
                did_update_support = true
            else
                xs, ps = marginal_density_cubes[node.address]
                l_prec, r_prec = largest_bounds[node.address]

                # first we try to find an interval such that the ends are below density_thresh
                if 0 < l_prec && l_prec < 10 || 0 < r_prec && r_prec < 10
                    if ps[1] > density_thresh
                        l_prec += 1 # change left quantile bound by order of magnitude
                    end
                    if ps[end] > density_thresh
                        r_prec += 1 # change right quantile bound by order of magnitude
                    end
                    if ps[1] < density_thresh && ps[end] < density_thresh
                        # we are finished with expanding interval -> set to 0
                        largest_bounds[node.address] = (0, 0)
                    else
                        node.support = get_initial_aqua_support(pgm, node, parents, N, 10.0^(-l_prec), 10.0^(-r_prec))
                        did_update_support = true
                        println(node.address)
                        println("old support: ", xs[1], " ... ", xs[end], " ", largest_bounds[node.address])
                        println("new support: ", node.support[1], " ... ", node.support[end], " ($l_prec, $r_prec)")
                        largest_bounds[node.address] = (l_prec, r_prec)
                        continue
                    end
                end

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


                xs_new = Vector{Float64}(LinRange(x0, x1, N))

                println(node.address)
                println("old support: ", xs[1], " ... ", xs[end])
                println("new support: ", xs_new[1], " ... ", xs_new[end])
                
                node.support = xs_new
                @assert length(node.support) >= 2
                did_update_support = did_update_support || 1 < i || j < length(xs)
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

function aqua_ve(pgm::PGM, N::Int)
    result = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
    largest_bounds = Dict{Symbol,Tuple{Int,Int}}()

    count = 0
    while true
        count += 1
        count > 100 && break

        variable_nodes, factor_nodes, did_update_support = get_aqua_factor_graph(pgm, N, result, largest_bounds)
        !did_update_support && break
        
        for node in variable_nodes
            f, Z = variable_elimination(pgm, variable_nodes, factor_nodes, [node.variable], :Greedy)
            Δ = node.support[2] - node.support[1] # is not observed variable_node
            result[node.address] =  (node.support, exp.(f.table) / (Z * Δ))
        end
    end

    return result
end
export aqua_ve