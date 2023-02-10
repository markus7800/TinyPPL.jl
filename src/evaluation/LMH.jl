
import ..TinyPPL.Distributions: Proposal, logpdf

mutable struct LMH <: Sampler
    W::Float64
    Q::Float64
    proposal::Proposal
    resample_addr::Any
    trace_current::Dict{Any, Real}
    trace::Dict{Any, Real}
    function LMH(proposal)
        return new(0., 0., proposal, nothing, Dict(), Dict())
    end
end

function sample(sampler::LMH, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    proposal_dist = get(sampler.proposal, addr, dist)
    if sampler.resample_addr == addr
        value = rand(proposal_dist)
    else
        value = get(sampler.trace_current, addr, rand(proposal_dist))
    end
    sampler.W += logpdf(dist, value)
    sampler.Q += logpdf(proposal_dist, value)
    sampler.trace[addr] = value
    
    return value
end

function lmh(model::Function, args::Tuple, observations::Dict, n_samples::Int, proposal::Proposal=Proposal())
    traces = Vector{Dict{Any, Real}}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LMH(proposal)

    n_accepted = 0
    retval_current = model(args..., sampler, observations)
    W_current = sampler.W
    Q_current = sampler.Q
    trace_current = sampler.trace
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.Q = 0.
        sampler.trace_current = trace_current
        sampler.trace = Dict{Any, Real}()
        sampler.resample_addr = rand(keys(sampler.trace_current))

        retval_proposed = model(args..., sampler, observations)
        trace_proposed = sampler.trace
        Q_proposed = sampler.Q
        W_proposed = sampler.W

        log_α = W_proposed - Q_proposed - W_current + Q_current + log(length(trace_current)) - log(length(trace_proposed))

        if log(rand()) < log_α
            retval_current, trace_current = retval_proposed, trace_proposed
            W_current, Q_current = W_proposed, Q_proposed
            n_accepted += 1
        end

        @inbounds retvals[i] = retval_current
        @inbounds logprobs[i] = W_current
        @inbounds traces[i] = trace_current
    end
    @info "LMH" n_accepted/n_samples

    return traces, retvals, logprobs
end

export lmh