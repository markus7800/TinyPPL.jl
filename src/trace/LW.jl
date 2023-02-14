
function likelihood_weighting(model::Function, args::Tuple, observations::Dict, n_samples::Int)
    traces = Trace[Trace() for _ in 1:n_samples]
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    @progress for i in 1:n_samples
        retvals[i], trace = model(args..., observations, traces[i])
        logprobs[i] = sum(rv.logprob for (addr,rv) in trace if haskey(observations, addr))
    end
    return traces, retvals, normalise(logprobs)
end

export likelihood_weighting