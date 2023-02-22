import ..TinyPPL.Distributions: logpdf

function likelihood_weighting(pgm::PGM, n_samples::Int)

    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)

    observed = .!isnothing.(pgm.observed_values)
    
    X = Vector{Float64}(undef, pgm.n_variables)
    @progress for i in 1:n_samples
        W = 0.
        for node in pgm.topological_order
            d = pgm.distributions[node](X)

            if observed[node]
                value = pgm.observed_values[node](X)
                W += logpdf(d, value)
            else
                value = rand(d)
            end
            X[node] = value
        end
        r = pgm.return_expr(X)

        logprobs[i] = W
        retvals[i] = r
        trace[:,i] = X
    end

    return trace, retvals, normalise(logprobs)
end

export likelihood_weighting
