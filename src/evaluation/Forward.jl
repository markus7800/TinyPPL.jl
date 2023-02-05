

mutable struct Forward <: Sampler
end


function sample(sampler::Forward, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end

    return rand(dist)
end

export Forward