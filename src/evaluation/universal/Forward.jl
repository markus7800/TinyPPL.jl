
"""
Samples values at sample statements if no observation is provided,
else uses the observed value.
Has no other side effects.
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
Samples values at sample statements if no observation is provided,
else uses the observed value.
Accumulates the log density of sample and observed values separately.
"""
mutable struct LogProb <: UniversalSampler
    log_p_Y::Float64 # log p(Y|X)
    log_p_X::Float64 # log p(X)
    function LogProb()
        return new(0., 0.)
    end
end


function sample(sampler::LogProb, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.log_p_Y += logpdf(dist, obs)
        return obs
    end

    value = rand(dist)
    sampler.log_p_X += logpdf(dist, value)
    return value
end

export LogProb