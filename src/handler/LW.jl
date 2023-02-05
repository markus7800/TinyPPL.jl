
function likelihood_weighting(model::Function, args::Tuple, n_samples::Int)
    traces = Vector{Trace}(undef, n_samples)
    retvals = Vector{Float64}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    @progress for i in 1:n_samples
        @inbounds traces[i] = get_trace(trace(model), args...)
        @inbounds retvals[i] = traces[i][:RETURN]["value"]
        @inbounds logprobs[i] = logpdfsum(traces[i], msg -> msg["type"] == "observation")
    end
    return traces, retvals, normalise(logprobs)
end

export likelihood_weighting