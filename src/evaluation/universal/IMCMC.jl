import LinearAlgebra: logabsdet

"""
Computes log p(X,Y) for trace X produced by an involution.
In contrast to TraceSampler, missing addresses are not sampled, since there would be no
corresponding proposal term in the acceptance probability. Throws an error instead.
Also records visited addresses, such that we can check that X contains no superfluous addresses.
"""
mutable struct StrictTraceScorer <: UniversalSampler
    W::Float64
    X::AbstractUniversalTrace
    visited::Set{Any}
    StrictTraceScorer(X::AbstractUniversalTrace) = new(0., X, Set{Any}())
end

function sample(sampler::StrictTraceScorer, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    if !haskey(sampler.X, addr)
        error("Address $addr is missing in trace produced by involution. The involution has to provide values for all addresses that the (auxiliary) model visits.")
    end
    push!(sampler.visited, addr)
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    return value
end

function strict_score_trace(model::UniversalModel, args::Tuple, observations::AbstractObservations, X::AbstractUniversalTrace)
    sampler = StrictTraceScorer(X)
    retval = model(args, sampler, observations)
    if length(sampler.visited) != length(X)
        superfluous = setdiff(keys(X), sampler.visited)
        error("Trace produced by involution contains addresses that are not visited by the (auxiliary) model: $superfluous")
    end
    return sampler.W, retval
end

"""
Involutive MCMC
extend model with auxilliary model p(v|x)p(x)

propose from auxilliary model v ~ p(v|x)
transform with involution f: (x',v') = f(x,v)
compute acceptance probability p(x',v') / p(x,v) * |det ∇f(x,v)|

involution has to be implemented according to the TraceTransformation protocol.

if check_involution = true, then we check at each iteration if passed function
is indeed involution for current proposed value.

Returns traces (like all MCMC algorithms).
"""
function imcmc(model::UniversalModel, args::Tuple, observations::AbstractObservations,
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

        # compute p(x'), x' has to be complete (no missing / superfluous addresses)
        W_proposed, retval_proposed = strict_score_trace(model, args, observations, trace_proposed)

        # compute p(v'|x')
        Q_proposed, _ = strict_score_trace(aux_model, (trace_proposed, aux_args...), aux_no_obs, aux_proposed)

        # perform MH step
        if log(rand()) < W_proposed + Q_proposed - W_current - Q_current + logabsdet(J)[1]
            n_accepted += 1
            trace_current = trace_proposed
            W_current = W_proposed
            retval_current = retval_proposed
        end

        traces[i] = trace_current
        retvals[i] = retval_current
    end
    
    @info "iMCMC" n_accepted/n_samples

    return UniversalTraces(traces, retvals)
end

export imcmc