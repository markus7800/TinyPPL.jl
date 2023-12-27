
import ..TinyPPL.Distributions: Proposal, logpdf, proposal_logpdf, propose_and_logpdf, StaticProposal
import Random

abstract type UniversalSingleSiteSampler <: UniversalSampler end

mutable struct LMH <: UniversalSingleSiteSampler
    W::Float64
    Q::Dict{Any, Float64}
    Q_resample_address::Float64
    proposal::Proposal
    resample_addr::Any
    trace_current::Dict{Any, Real}
    trace::Dict{Any, Real}
    function LMH(proposal)
        return new(0., Dict{Any, Float64}(), 0., proposal, nothing, Dict{Any, Real}(), Dict{Any, Real}())
    end
end

function sample(sampler::LMH, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    proposal_dist = get(sampler.proposal, addr, StaticProposal(dist))
    if sampler.resample_addr == addr
        x_current = sampler.trace_current[addr]
        x_proposed, forward_logpdf = propose_and_logpdf(proposal_dist, x_current)
        backward_logpdf = proposal_logpdf(proposal_dist, x_current, x_proposed)
        
        # current - proposed
        sampler.Q_resample_address += backward_logpdf - forward_logpdf
        value = x_proposed
    else
        # if we don't have previous value to move from, sample from prior
        value = get(sampler.trace_current, addr, rand(dist))
        # value = get(sampler.trace_current, addr, rand(proposal_dist)) # could check if proposal dist is static
    end
    sampler.W += logpdf(dist, value)
    sampler.Q[addr] = logpdf(dist, value)
    # sampler.Q[addr] = logpdf(proposal_dist, value) # could check if proposal dist is static
    sampler.trace[addr] = value
    return value
end

function single_site_sampler(model::UniversalModel, args::Tuple, observations::Dict, n_samples::Int, sampler::UniversalSingleSiteSampler, gibbs::Bool)
    traces = Vector{Dict{Any, Real}}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)

    n_accepted = 0
    retval_current = model(args, sampler, observations)
    W_current = sampler.W
    Q_current = sampler.Q
    trace_current = sampler.trace

    addresses = gibbs ? collect(keys(trace_current)) : Any[nothing]

    @progress for i in 1:n_samples
        if gibbs
            Random.shuffle!(addresses)
        else
            addresses[1] = rand(keys(trace_current))
        end
        for addr in addresses
            sampler.W = 0.
            sampler.Q = Dict{Any, Float64}()
            sampler.Q_resample_address = 0.
            sampler.trace_current = trace_current
            sampler.trace = Dict{Any, Real}()

            sampler.resample_addr = addr

            retval_proposed = model(args, sampler, observations)
            trace_proposed = sampler.trace
            Q_proposed = sampler.Q
            W_proposed = sampler.W

            sampler.Q_resample_address += log(length(trace_current)) - log(length(trace_proposed))

            log_α = W_proposed - W_current + sampler.Q_resample_address
            for (addr, q) in Q_proposed
                if !haskey(Q_current, addr) # resampled
                    log_α -= q
                end
            end
            for (addr, q) in Q_current
                if !haskey(Q_proposed, addr) # resampled
                    log_α += q
                end
            end

            if log(rand()) < log_α
                retval_current, trace_current = retval_proposed, trace_proposed
                W_current, Q_current = W_proposed, Q_proposed
                n_accepted += 1
            end
        end

        retvals[i] = retval_current
        logprobs[i] = W_current
        traces[i] = trace_current
    end
    @info "SingleSite $(typeof(sampler))" n_accepted/(n_samples*length(addresses))

    return traces, retvals, logprobs
end

function lmh(model::UniversalModel, args::Tuple, observations::Dict, n_samples::Int; 
    proposal::Proposal=Proposal(), gibbs::Bool=false)

    sampler = LMH(proposal)
    return single_site_sampler(model, args, observations, n_samples, sampler, gibbs)
end


export single_site_sampler, lmh