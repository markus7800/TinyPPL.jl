

import ..TinyPPL.Distributions: logpdf, random_walk_proposal_dist, Addr2Var

"""
Single-Site Random Walk Metropolis Hastings
Equivalent to LMH with the difference that the conditional proposal at the
resample site is a Random Walk proposal given in distributions/random_walk.jl
The user may specifiy the variance of each random walk proposal kernel and 
a default variance.
"""
mutable struct RWMH <: UniversalSingleSiteSampler
    W::Float64                      # log p(X,Y)
    Q::Dict{Any,Float64}            # proposal density
    trace_current::Trace            # current trace X
    trace_proposed::Trace           # proposed trace X'
    resample_addr::Any              # address X0 at which to resample value
    Q_resample_address::Float64     # log Q(x_current | x_proposed) - log Q(x_proposed | x_current)
    default_var::Float64            # default random walk variance
    addr2var::Addr2Var              # mapping of address to random walk variance
    function RWMH(default_var::Float64, addr2var::Addr2Var)
        return new(0., Dict{Any, Float64}(), Trace(), Trace(),  nothing, 0., default_var, addr2var)
    end
end

function sample(sampler::RWMH, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        # accumulated log density p(X,Y)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    if sampler.resample_addr == addr
        # propose from random walk proposal distribution conditional on current value
        value_current = sampler.trace_current[addr]
        var = get(sampler.addr2var, addr, sampler.default_var)
        forward_dist = random_walk_proposal_dist(dist, value_current, var)
        value = rand(forward_dist) # propsed value

        backward_dist = random_walk_proposal_dist(dist, value, var)
        sampler.Q_resample_address += logpdf(backward_dist, value_current) - logpdf(forward_dist, value)
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

# This is equivalent to running lmh with setting every proposal distribution
# to the corresponding RandomWalkProposal competible with addr2var
function rwmh(model::UniversalModel, args::Tuple, observations::Dict, n_samples::Int;
    default_var::Float64=1., addr2var::Addr2Var=Addr2Var(), gibbs=false)
    sampler = RWMH(default_var, addr2var)
    return single_site_sampler(model, args, observations, n_samples, sampler, gibbs)
end

export rwmh