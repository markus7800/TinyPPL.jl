
function compiled_single_site(pgm::PGM, kernels::Vector{Function}, n_samples::Int; static_observes::Bool=false)

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample!(X) # initialise
    r = pgm.return_expr(X)

    mask = isnothing.(pgm.observed_values)
    trace = Array{Float64,2}(undef, static_observes ? sum(mask) : pgm.n_variables, n_samples)
    retvals = Vector{Any}(undef, n_samples)

    n_accepted = 0 
    @progress for i in 1:n_samples
        k = rand(kernels)
        accepted = k(X)
        if accepted
            n_accepted += 1
            r = pgm.return_expr(X)
        end

        retvals[i] = r
        if static_observes
            trace[:,i] = X[mask]
        else  
            trace[:,i] = X
        end
    end
    @info "Compiled Single Site" n_accepted/n_samples

    return trace, retvals
end

function get_compute_W_block_args(pgm, plates, children, symbolic_dists, symbolic_observes, static_observes, X, log_α, W)
    block_args = []
    for child in children
        if child in plates
            plate_f_name = plate_function_name(pgm.name, :lp, child)
            if W == :current
                push!(block_args, :($log_α -= $plate_f_name($X)))
            else
                push!(block_args, :($log_α += $plate_f_name($X)))
            end
        else
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(symbolic_observes[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            if W == :current
                push!(block_args, :($log_α -= logpdf($child_d_sym, $X[$child])))
            else
                push!(block_args, :($log_α += logpdf($child_d_sym, $X[$child])))
            end
        end
    end
    return block_args
end

function compile_lmh(pgm::PGM; static_observes::Bool=false, proposal=Proposal())
    function lmh_kernel(block_args, symbolic_dists, node, X, log_α)
        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $(symbolic_dists[node])))

        if haskey(proposal, pgm.addresses[node])
            q_sym = gensym("proposal_$node")
            q = proposal[pgm.addresses[node]]
            push!(block_args, :($q_sym = $(Expr(:call, typeof(q).name.name, params(q)...))))

            push!(block_args, :($log_α += logpdf($q_sym, $X[$node]) - logpdf($d_sym, $X[$node])))
            push!(block_args, :($X[$node] = rand($q_sym)))
            push!(block_args, :($log_α += logpdf($d_sym, $X[$node]) - logpdf($q_sym, $X[$node])))
        else
            # logpdf(d, value_proposed) - logpdf(d, value_current) +  logpdf(q, value_current) - logpdf(q, value_proposed) cancels
            push!(block_args, :($X[$node] = rand($d_sym)))
        end
    end
    return compile_single_site(pgm, static_observes, lmh_kernel)
end

function get_rw_dist_expr(d_sym, symbol_dist, value, var)
    if symbol_dist.head == :call
        dist_name = symbol_dist.args[1]
        # println(dist_name)
        if dist_name == :Bernoulli
            return :(Bernoulli(1 - $value))
        elseif dist_name in [:Geometric, :Poisson]
            return :(DiscreteRWProposer(0, Inf, Int($value), $var))
        elseif dist_name == :Binomial
            n = symbol_dist.args[2]
            return :(DiscreteRWProposer(0, $n, Int($value), $var))
        elseif dist_name == :Categorical
            return :(DiscreteRWProposer(1, ncategories($d_sym), Int($value), $var))
        elseif dist_name == :DiscreteUniform
            a, b = symbol_dist.args[2:3]
            return :(DiscreteRWProposer(Int($a), Int($b), Int($value), $var))
        elseif dist_name in [:Exponential, :Gamma, :InverseGamma, :LogNormal]     
            return :(ContinuousRWProposer(0, Inf, $value, $var))
        elseif dist_name in [:Cauchy, :Laplace, :Normal, :TDist]      
            return :(Normal($value, sqrt($var)))  
        elseif dist_name == :Beta
            return :(ContinuousRWProposer(0., 1., $value, $var))
        elseif dist_name == :Uniform
            a, b = symbol_dist.args[2:3]
            return :(ContinuousRWProposer($a, $b, $value, $var))
        end
    end
    return :(random_walk_proposal_dist($d_sym, $value, $var))
end

function compile_rwmh(pgm::PGM; static_observes::Bool=false, addr2var=Addr2Var(), default_var::Float64=1.)
    function rwmh_kernel(block_args, symbolic_dists, node, X, log_α)
        dist = symbolic_dists[node]
        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $dist))

        var = get(addr2var, pgm.addresses[node], default_var)

        forward_d_sym = gensym("forward_dist")
        value_current = gensym("value_current")
        backward_d_sym = gensym("backward_dist")
        value_proposed = gensym("value_proposed")


        push!(block_args, :($value_current = $X[$node]))
        push!(block_args, :($forward_d_sym = $(get_rw_dist_expr(d_sym, dist, value_current, var))))
        push!(block_args, :($value_proposed = rand($forward_d_sym)))
        push!(block_args, :($X[$node] = $value_proposed))
        push!(block_args, :($backward_d_sym = $(get_rw_dist_expr(d_sym, dist, value_proposed, var))))

        push!(block_args, :($log_α += logpdf($backward_d_sym, $value_current) - logpdf($d_sym, $value_current)))
        push!(block_args, :($log_α += logpdf($d_sym, $value_proposed) - logpdf($forward_d_sym, $value_proposed)))
    end
    return compile_single_site(pgm, static_observes, rwmh_kernel)
end

function compile_single_site(pgm::PGM, static_observes::Bool, kernel::Function)
    X = gensym(:X)
    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)

    if isnothing(pgm.plate_info)
        plates, plated_edges = Plate[], nothing
    else
        plates, plated_edges = pgm.plate_info.plates, pgm.plate_info.plated_edges
    end

    lmh_functions = Function[]
    for node in 1:pgm.n_variables
        !isnothing(pgm.observed_values[node]) && continue

        block_args = []
        value_current = gensym("value_current")
        push!(block_args, :($value_current = $X[$node]))

        if isnothing(plated_edges)
            children = [child for (x,child) in pgm.edges if x == node]
        else
            children = reduce(∪, [get_children(edge, node) for edge in plated_edges], init=[])
        end
 
        log_α = gensym(:log_α) # W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        push!(block_args, :($log_α = 0.0))

        # compute W for current value
        append!(block_args, 
            get_compute_W_block_args(pgm, plates, children, symbolic_dists, symbolic_observes, static_observes, X, log_α, :current)
        )

        # sample proposed value
        kernel(block_args, symbolic_dists, node, X, log_α)

        # compute W for proposed value
        append!(block_args, 
            get_compute_W_block_args(pgm, plates, children, symbolic_dists, symbolic_observes, static_observes, X, log_α, :proposed)
        )

        # mh step
        push!(block_args, :(if log(rand()) < $log_α
                return true
            else
                $X[$node] = $value_current
                return false
            end)
        )  

        f_name = Symbol("$(pgm.name)_lmh_$node")
        f = rmlines(:(
            function $f_name($X::Vector{Float64})
                $(Expr(:block, block_args...))
            end
        ))
        # display(f)
        f = eval(f)
        push!(lmh_functions, f)
    end

    X = Vector{Float64}(undef, pgm.n_variables);
    pgm.sample!(X) # initialise
    for f in lmh_functions
        # println(f)
        Base.invokelatest(f, X)
    end
    return lmh_functions
end

export compile_lmh, compile_rwmh, compiled_single_site