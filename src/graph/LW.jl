import Distributions: logpdf

function likelihood_weighting(pgm::PGM, n_samples::Int)

    retvals = Vector{Float64}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)
    
    X = Vector{Float64}(undef, pgm.n_variables)
    @progress for i in 1:n_samples
        W = 0.
        for node in pgm.topological_order
            d = pgm.distributions[node](X)

            if !isnothing(pgm.observed_values[node])
                value = pgm.observed_values[node](X)
                X[node] = value
                W += logpdf(d, value)
            else
                value = rand(d)
                X[node] = value
            end
        end
        r = pgm.return_expr(X)

        @inbounds logprobs[i] = W
        @inbounds retvals[i] = r
        @inbounds trace[:,i] .= X
        
    end

    return trace, retvals, normalise(logprobs)
end

export likelihood_weighting