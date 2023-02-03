

mutable struct LW <: Sampler
    W::Float64
    function LW()
        return new(0.)
    end
end

function sample(sampler::LW, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    return rand(dist)
end

function likelihood_weighting(model::Function, args::Tuple, observations::Dict, n_samples::Int)
    retvals = Vector{Float64}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        @inbounds retvals[i] = model(args..., sampler, observations)
        @inbounds logprobs[i] = sampler.W
    end
    return retvals, normalise(logprobs)
end

export LW, likelihood_weighting