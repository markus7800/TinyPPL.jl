import LinearAlgebra: logabsdet

"""
Involutive MCMC
extend model with auxilliary model p(v|x)p(x)

propose from auxilliary model v ~ p(v|x)
transform with involution f: (x',v') = f(x,v)
compute acceptance probability p(x',v') / p(x,v) * |det ∇f(x,v)|

involution has to be implemented according to the TraceTransformation protocol.

if check_involution = true, then we check at each iteration if passed function
is indeed involution for current proposed value.
"""
function imcmc(model::UniversalModel, args::Tuple, observations::Observations,
    aux_model::UniversalModel, aux_args::Tuple, involution::Function,
    n_samples::Int; check_involution::Bool=false)

    transformation = TraceTransformation(involution)
    sampler = TraceSampler()
    aux_no_obs = Observations()

    # initialse
    retval_current = model(args, sampler, observations)
    trace_current = sampler.X
    W_current = sampler.W

    traces = Vector{UniversalTrace}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    lps = Vector{Float64}(undef, n_samples) # log p(X,Y)

    n_accepted = 0
    @progress for i in 1:n_samples
        # propose from auxilliary v ~ p(v|x)
        sampler.W = 0.; sampler.X = UniversalTrace()
        aux_model((trace_current, aux_args...), sampler, aux_no_obs)
        aux_current = sampler.X
        Q_current = sampler.W

        # apply involution and compute jacobian
        trace_proposed, aux_proposed = apply(transformation, trace_current, aux_current)
        J = jacobian_fwd_diff(transformation, trace_current, aux_current, trace_proposed, aux_proposed)

        # check involution if wanted
        if check_involution
            @assert size(J,1) == size(J,2)
            trace_current_2, aux_current_2 = apply(transformation, trace_proposed, aux_proposed)
            for addr in keys(trace_current) ∪ keys(trace_current_2)
                @assert trace_current[addr] ≈ trace_current_2[addr] (addr, trace_current[addr], trace_current_2[addr])
            end 
            for addr in keys(aux_current) ∪ keys(aux_current_2)
                @assert aux_current[addr] ≈ aux_current_2[addr] (addr, aux_current[addr], aux_current_2[addr])
            end
        end

        # compute p(x')
        sampler.W = 0.; sampler.X = trace_proposed
        retval_proposed = model(args, sampler, observations)
        W_proposed = sampler.W

        # compute p(v'|x')
        sampler.W = 0.; sampler.X = aux_proposed
        aux_model((trace_proposed, aux_args...), sampler, aux_no_obs)
        Q_proposed = sampler.W

        # perform MH step
        if log(rand()) < W_proposed + Q_proposed - W_current - Q_current + logabsdet(J)[1]
            n_accepted += 1
            trace_current = trace_proposed
            W_current = W_proposed
            retval_current = retval_proposed
        end

        traces[i] = trace_current
        lps[i] = W_current
        retvals[i] = retval_current
    end
    
    @info "iMCMC" n_accepted/n_samples

    return UniversalTraces(traces, retvals), lps
end

export imcmc