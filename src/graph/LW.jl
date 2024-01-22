import ..TinyPPL.Distributions: logpdf

function likelihood_weighting(pgm::PGM, n_samples::Int)

    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_latents, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    
    X = Vector{Float64}(undef, pgm.n_latents)
    @progress for i in 1:n_samples
        W = 0.
        for node in pgm.topological_order
            d = get_distribution(pgm, node, X)

            if isobserved(pgm, node)
                value = get_observed_value(pgm, node)
                W += logpdf(d, value)
            else
                value = rand(d)
                X[node] = value
            end
        end
        r = get_retval(pgm, X)

        logprobs[i] = W
        retvals[i] = r
        trace[:,i] = X
    end

    return GraphTraces(pgm, trace, retvals), normalise(logprobs)
end

export likelihood_weighting
