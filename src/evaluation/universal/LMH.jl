
import ..TinyPPL.Distributions: logpdf, proposal_logpdf, propose_and_logpdf, Addr2Proposal, StaticProposal
import Random

abstract type UniversalSingleSiteSampler <: UniversalSampler end

"""
Lightweight Metropolis Hastings

Idea: change value (resample) of only at one address in current trace
Challenge: this may affect which other addresses are missing or new in proposed trace.

The user can define the conditional update for the resample address Q(x_proposed | x_current)
by providing a ProposalDistribution with Addr2Proposal.
Otherwise, by default we unconditionally propose from the program prior.

The acceptance probability is given by

log α = log p(X',Y') + log Q(X|X') - log p(X,Y) - log Q(X'|X), where

log Q(X'|X) = log(1/|X|) + log Q(x_proposed | x_current) + sum(log Q[addr] for addr in addresses(X') - addresses(X))

addresses(X') - addresses(X) are all the addresses that need to be newly sampled when going from X to X'.
"""
mutable struct LMH <: UniversalSingleSiteSampler
    W::Float64                      # log p(X,Y)
    Q::Dict{Any,Float64}            # proposal density
    trace_current::UniversalTrace   # current trace X
    trace_proposed::UniversalTrace  # proposed trace X'
    resample_addr::Any              # address X0 at which to resample value
    Q_resample_address::Float64     # log Q(x_current | x_proposed) - log Q(x_proposed | x_current)
    addr2proposal::Addr2Proposal    # map of addresses to proposal distribution
    function LMH(addr2proposal::Addr2Proposal)
        return new(0., Dict{Any,Float64}(), UniversalTrace(),UniversalTrace(), nothing, 0., addr2proposal)
    end
end

function sample(sampler::LMH, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        # accumulated log density p(X,Y)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    # by default propose from prior
    proposal_dist = get(sampler.addr2proposal, addr, StaticProposal(dist))

    if sampler.resample_addr == addr
        # propose from proposal distribution conditional on current value
        x_current = sampler.trace_current[addr]
        x_proposed, forward_logpdf = propose_and_logpdf(proposal_dist, x_current)
        backward_logpdf = proposal_logpdf(proposal_dist, x_current, x_proposed)
        
        sampler.Q_resample_address += backward_logpdf - forward_logpdf
        value = x_proposed
    else
        # if we don't have previous value to move from, sample from prior
        value = get(sampler.trace_current, addr, rand(dist))
    end

    # accumulated log density p(X,Y)
    sampler.W += logpdf(dist, value)

    # we need to store the proposal density (sampling from prior) for every address,
    # since we do not know which addresses are missing from reference trace 
    sampler.Q[addr] = logpdf(dist, value)
    sampler.trace_proposed[addr] = value

    return value
end

# if gibbs = true all addresses in current trace will be resampled in random order in one iteration
# this is usefull for static models
# else only one resample address is chosen randomly per iteraion
# this is worker functoin which may be used by any UniversalSingleSiteSampler (LMH or RWMH)
# returns samples X_i ~ p(X|Y) 
# returns also `logprobs` log p(X,Y) = log p(X|Y) + log p(Y), which my be used to estimate MAP
function single_site_sampler(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int, sampler::UniversalSingleSiteSampler, gibbs::Bool)
    traces = Vector{UniversalTrace}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples) # log p(X,Y)

    # initialise
    retval_current = model(args, sampler, observations)
    W_current = sampler.W
    Q_current = sampler.Q
    trace_current = sampler.trace_proposed

    addresses = gibbs ? collect(keys(trace_current)) : Any[nothing]

    n_accepted = 0
    @progress for i in 1:n_samples
        if gibbs
            Random.shuffle!(addresses)
        else
            addresses[1] = rand(keys(trace_current))
        end
        for addr in addresses # if gibbs = false only one address

            # reset sampler
            sampler.W = 0.
            sampler.Q = Dict{Any, Float64}()
            sampler.Q_resample_address = 0.
            sampler.trace_current = trace_current
            sampler.trace_proposed = UniversalTrace()

            # set resample address
            sampler.resample_addr = addr

            # run model with sampler
            retval_proposed = model(args, sampler, observations)
            trace_proposed = sampler.trace_proposed
            Q_proposed = sampler.Q
            W_proposed = sampler.W

            # compute acceptance probability
            log_α = W_proposed - W_current + sampler.Q_resample_address + log(length(trace_current)) - log(length(trace_proposed))
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

            # accept or reject
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

    return UniversalTraces(traces, retvals), logprobs
end

function lmh(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int; 
    addr2proposal::Addr2Proposal=Addr2Proposal(), gibbs::Bool=false)

    sampler = LMH(addr2proposal)
    return single_site_sampler(model, args, observations, n_samples, sampler, gibbs)
end


export single_site_sampler, lmh