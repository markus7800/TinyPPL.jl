
"""
Samples values at sample statements if no observation is provided,
else uses the observed value.
Has no other side effects.
This is the most basic sampler.
"""
mutable struct Forward <: UniversalSampler
end


function sample(sampler::Forward, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end

    return rand(dist)
end

export Forward


"""
Samples values at sample statements if no value is provided with `X`
Accumulates the log density of sample and observed values.
Can be used to sample with score from model, or to evaluate score of given `X`.
"""
mutable struct TraceSampler <: UniversalSampler
    W::Float64                  # log p(X,Y)
    X::AbstractUniversalTrace   # trace to evaluate log p at
    function TraceSampler(X::AbstractUniversalTrace=UniversalTrace())
        return new(0., X)
    end
end

function sample(sampler::TraceSampler, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    # evaluate at given value or sample and store
    if !haskey(sampler.X, addr)
        sampler.X[addr] = rand(dist)
    end
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    return value
end

"""
Convenience method for sampling from model.
"""
function sample_trace(model::UniversalModel, args::Tuple, observations::Observations=Observations())
    sampler = TraceSampler()
    model(args, sampler, observations)
    return sampler.X
end
export sample_trace

"""
Convenience method for computing log p(X,Y) for given trace `X` according to model.
"""
function score_trace(model::UniversalModel, args::Tuple, observations::Observations, X::AbstractUniversalTrace)
    sampler = TraceSampler(X)
    model(args, sampler, observations)
    return sampler.W
end
function score_trace(model::UniversalModel, args::Tuple, X::AbstractUniversalTrace)
    return score_trace(model, args, Observations(), X)
end
export score_trace