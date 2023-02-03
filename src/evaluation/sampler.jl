
abstract type Sampler end

function sample(sampler::Sampler, dist::Distribution, obs::Union{Nothing, Real})::Real
    error("sample not implemented for $sampler.")
end