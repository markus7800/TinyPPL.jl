
"""
Samples values at sample statement and stores them in trace if no observed value is provided,
else uses the observed value and accumulates its log density in `W`.
"""
mutable struct LW <: UniversalSampler
    W::Float64 # log p(Y|X)
    trace::Trace
    function LW()
        return new(0., Trace())
    end
end

function sample(sampler::LW, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    value = rand(dist)
    sampler.trace[addr] = value
    
    return value
end

"""
Likelihood weighting is a special case of importance sampling where the reference distribution is
the program prior p(X).
We repeatedly sample from the program prior X_i ~ p(X) and record the Likelihood w_i = p(Y|X_i).
The posterior can be approximated with ∑ w̃_i δ_{X_i},
where w̃_i are the normalised weights w̃_i = w_i / ∑ w_i
"""
function likelihood_weighting(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int)
    traces = Vector{Trace}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = Trace()
        retvals[i] = model(args, sampler, observations)
        logprobs[i] = sampler.W
        traces[i] = sampler.trace
    end
    return UniversalTraces(traces, retvals), normalise(logprobs)
end

function no_op_completion(trace::Trace, retval::Any)
    return nothing
end

"""
The same behavior as likelihood weighting but you may specify a completion handler,
which takes for each iteration the trace and return value and maps it to an return type.
This may be used if you do not care about the entire trace of a program and want
to save memory.
"""
function likelihood_weighting(model::UniversalModel, args::Tuple, observations::Dict, n_samples::Int, completion::Function)
    result =  Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = empty!(sampler.trace)
        retval = model(args, sampler, observations)
        logprobs[i] = sampler.W
        result[i] = completion(sampler.trace, retval)
    end
    return result, normalise(logprobs)
end

export likelihood_weighting