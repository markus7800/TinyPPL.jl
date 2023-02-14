
function likelihood_weighting(model::Function, args::Tuple, n_samples::Int)
    traces = Vector{Trace}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    @progress for i in 1:n_samples
        traces[i] = get_trace(trace(model), args...)
        retvals[i] = traces[i][:RETURN]["value"]
        logprobs[i] = logpdfsum(traces[i], msg -> msg["type"] == "observation")
    end
    return traces, retvals, normalise(logprobs)
end

export likelihood_weighting