

mutable struct LW <: UniversalSampler
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

function likelihood_weighting(model::UniversalModel, args::Tuple, observations::Dict, n_samples::Int)
    traces = Vector{Dict{Any, Real}}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = Dict{Any, Real}()
        retvals[i] = model(args, sampler, observations)
        logprobs[i] = sampler.W
        traces[i] = sampler.trace
    end
    return traces, retvals, normalise(logprobs)
end

function no_op_completion(trace::Dict{Any, Real})
    return nothing
end

function likelihood_weighting(model::UniversalModel, args::Tuple, observations::Dict, n_samples::Int, completion::Function)
    result =  Vector{Any}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = empty!(sampler.trace)
        retvals[i] = model(args, sampler, observations)
        logprobs[i] = sampler.W
        result[i] = completion(sampler.trace)
    end
    return result, retvals, normalise(logprobs)
end

export likelihood_weighting