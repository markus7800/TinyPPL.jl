

mutable struct LW <: Sampler
    W::Float64
    trace::Dict{Any, Real}
    function LW()
        return new(0., Dict())
    end
end

function sample(sampler::LW, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    value = rand(dist)
    sampler.trace[addr] = value
    
    return value
end

function likelihood_weighting(model::Function, args::Tuple, observations::Dict, n_samples::Int)
    traces = Vector{Dict{Any, Real}}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = Dict{Any, Real}()
        @inbounds retvals[i] = model(args..., sampler, observations)
        @inbounds logprobs[i] = sampler.W
        @inbounds traces[i] = sampler.trace
    end
    return traces, retvals, normalise(logprobs)
end

export likelihood_weighting