

mutable struct Forward <: Sampler
end


function sample(sampler::Forward, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end

    return rand(dist)
end

export Forward


mutable struct LogProb <: Sampler
    log_p_Y::Float64
    log_p_X::Float64
    function LogProb()
        return new(0., 0.)
    end
end


function sample(sampler::LogProb, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.log_p_Y += logpdf(dist, obs)
        return obs
    end

    value = rand(dist)
    sampler.log_p_X += logpdf(dist, value)
    return value
end

export LogProb