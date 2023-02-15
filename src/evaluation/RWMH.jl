

import ..TinyPPL.Distributions: Addr2Var, logpdf, random_walk_proposal_dist

mutable struct RWMH <: Sampler
    W::Float64
    Q::Float64
    Q_correction::Float64
    default_var::Float64
    addr2var::Addr2Var
    resample_addr::Any
    trace_current::Dict{Any, Real}
    trace::Dict{Any, Real}
    function RWMH(default_var, addr2var)
        return new(0., 0., 0., default_var, addr2var, nothing, Dict(), Dict())
    end
end

function sample(sampler::RWMH, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    if sampler.resample_addr == addr
        value_current = sampler.trace_current[addr]
        var = get(sampler.addr2var, addr, sampler.default_var)
        forward_dist = random_walk_proposal_dist(dist, value_current, var)
        value = rand(forward_dist)
        backward_dist = random_walk_proposal_dist(dist, value, var)
        sampler.Q_correction += logpdf(forward_dist, value) - logpdf(backward_dist, value_current)
    else
        # if we don't have previous value to move from, sample from prior
        value = get(sampler.trace_current, addr, rand(dist))
    end
    sampler.W += logpdf(dist, value)
    sampler.Q += logpdf(dist, value)
    sampler.trace[addr] = value
    
    return value
end

function rwmh(model::Function, args::Tuple, observations::Dict, n_samples::Int;
    default_var::Float64=1., addr2var::Addr2Var=Addr2Var())
    sampler = RWMH(default_var, addr2var)
    return single_site_sampler(model, args, observations, n_samples, sampler)
end

export rwmh