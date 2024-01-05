
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

function sample(sampler::TraceSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    # evaluate at given value or sample and store
    value = get!(sampler.X, addr, rand(dist))
    sampler.W += logpdf(dist, value)
    return value
end