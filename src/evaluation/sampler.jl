
abstract type Sampler end

function sample(sampler::Sampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    error("sample not implemented for $sampler.")
end

abstract type StaticSampler <: Sampler end
abstract type UniversalSampler <: Sampler end

export sample