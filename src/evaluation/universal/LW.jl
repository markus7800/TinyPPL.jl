
"""
Samples values at sample statement and stores them in trace if no observed value is provided,
else uses the observed value and accumulates its log density in `W`.
"""
mutable struct LW <: UniversalSampler
    W::Float64 # log p(Y|X)
    trace::UniversalTrace
    function LW()
        return new(0., UniversalTrace())
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
Returns traces, normalised log weights log w̃_i and the log evidence estimate log(1/N ∑ w_i).

With `completion = (trace, retval) -> result`, the results of the completion are returned instead of traces,
see `likelihood_weighting_completion`.
"""
function likelihood_weighting(model::UniversalModel, args::Tuple, observations::AbstractObservations, n_samples::Int;
    completion::Union{Nothing,Function}=nothing)
    if !isnothing(completion)
        return likelihood_weighting_completion(model, args, observations, n_samples, completion)
    end
    traces = Vector{UniversalTrace}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = UniversalTrace()
        retvals[i] = model(args, sampler, observations)
        logprobs[i] = sampler.W
        traces[i] = sampler.trace
    end
    return UniversalTraces(traces, retvals), normalise_with_log_evidence(logprobs)...
end


"""
Same as likelihood_weighting, but we do not need weights.
"""
function sample_from_prior(model::UniversalModel, args::Tuple, observations::AbstractObservations, n_samples::Int)
    return likelihood_weighting(model, args, observations, n_samples)[1]
end
export sample_from_prior

function no_op_completion(trace::UniversalTrace, retval::Any)
    return nothing
end

function retval_completion(trace::UniversalTrace, retval::Any)
    return retval
end

"""
The same behavior as likelihood weighting but you may specify a completion handler,
which takes for each iteration the trace and return value and maps it to an return type.
This may be used if you do not care about the entire trace of a program and want
to save memory.
Returns the completion results, normalised log weights and the log evidence estimate.
"""
function likelihood_weighting_completion(model::UniversalModel, args::Tuple, observations::AbstractObservations, n_samples::Int, completion::Function)
    result =  Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = LW()
    @progress for i in 1:n_samples
        sampler.W = 0.
        # fresh trace for each sample, since the completion may keep the trace
        # (traces that are not kept by the completion are garbage collected)
        sampler.trace = UniversalTrace()
        retval = model(args, sampler, observations)
        logprobs[i] = sampler.W
        result[i] = completion(sampler.trace, retval)
    end
    return result, normalise_with_log_evidence(logprobs)...
end

export likelihood_weighting