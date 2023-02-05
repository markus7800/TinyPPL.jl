
abstract type Sampler end

function sample(sampler::Sampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    error("sample not implemented for $sampler.")
end