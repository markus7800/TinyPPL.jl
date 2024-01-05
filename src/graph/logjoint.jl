import Tracker
import TinyPPL.Distributions: mean

function make_logjoint(pgm::PGM)
    sample_mask = isnothing.(pgm.observed_values)
    n_sample = sum(sample_mask)
    @assert all(sample_mask[1:n_sample])
    @assert !any(sample_mask[n_sample+1:end])

    Z = Vector{Float64}(undef, pgm.n_variables)
    sample_mask = trues(pgm.n_variables)
    for node in pgm.topological_order
        d = pgm.distributions[node](Z)
        if !isnothing(pgm.observed_values[node])
            Z[node] = pgm.observed_values[node](Z) # observed values cannot depend on sampled values
        else
            Z[node] = mean(d)
        end
    end
    Y = Z[n_sample+1:end]

    function logjoint(X::AbstractVector{<:Real})
        X = convert(Vector{eltype(X)}, X) # tracked Vector -> Vector{Tracked}
        Z = vcat(X,Y)
        return pgm.unconstrained_logpdf!(Z)
    end
    return logjoint
end