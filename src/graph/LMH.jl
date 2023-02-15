
import ..TinyPPL.Distributions: Proposal, logpdf

function lmh(pgm::PGM, n_samples::Int; proposal=Proposal())
    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)

    observed = .!isnothing.(pgm.observed_values)

    nodes = [n => [child for (x,child) in pgm.edges if x == n] for n in pgm.topological_order if !observed[n]]

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample(X) # initialise
    r = pgm.return_expr(X)

    n_accepted = 0 
    @progress for i in 1:n_samples
        node, children = rand(nodes)
        d = pgm.distributions[node](X)
        q = get(proposal, pgm.addresses[node], d)
        value_current = X[node]
        # lp_current = pgm.logpdf(X)
        W_current = sum(logpdf(pgm.distributions[child](X), X[child]) for child in children) + logpdf(d, value_current)

        value_proposed = rand(q)
        X[node] = value_proposed

        # lp_proposed = pgm.logpdf(X)
        W_proposed = sum(logpdf(pgm.distributions[child](X), X[child]) for child in children) + logpdf(d, value_proposed)
        
        log_α = W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        # log_α = lp_proposed - lp_current + logpdf(d, value_current) - logpdf(d, value_proposed)

        if log(rand()) < log_α
            n_accepted += 1
            r = pgm.return_expr(X)
        else
            X[node] = value_current
        end

        retvals[i] = r
        trace[:,i] = X
    end
    @info "LMH" n_accepted/n_samples

    return trace, retvals
end

function compile_lmh(pgm::PGM; static_observes::Bool=false, proposal=Proposal())
    X = gensym(:X)
    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)
    
    lmh_functions = Function[]
    for node in 1:pgm.n_variables
        !isnothing(pgm.observed_values[node]) && continue

        block_args = []
        value_current = gensym("value_current")
        push!(block_args, :($value_current = $X[$node]))

        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $(symbolic_dists[node])))

        children = [child for (x,child) in pgm.edges if x == node]
 
        log_α = gensym(:log_α) # W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        push!(block_args, :($log_α = 0.0))

        # compute W for current value
        for child in children
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(pgm.observed_values[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            push!(block_args, :($log_α -= logpdf($child_d_sym, $X[$child])))
        end

        # sample proposed value
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

        # compute W for proposed value
        for child in children
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(pgm.observed_values[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            push!(block_args, :($log_α += logpdf($child_d_sym, $X[$child])))          
        end

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
            function $f_name($X::AbstractVector{Float64})
                $(Expr(:block, block_args...))
            end
        ))
        # display(f)
        push!(lmh_functions, eval(f))
    end

    X = Vector{Float64}(undef, pgm.n_variables);
    pgm.sample(X) # initialise
    for f in lmh_functions
        Base.invokelatest(f, X)
    end
    return lmh_functions
end

function compiled_single_site(pgm::PGM, kernels::Vector{Function}, n_samples::Int; static_observes::Bool=false)

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample(X) # initialise
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

export lmh, compile_lmh, compiled_single_site