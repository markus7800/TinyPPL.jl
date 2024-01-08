import TinyPPL.Logjoint: hmc_logjoint

"""
The hmc methods for static models also calls the logjoint implementation.
We sample the unconstrained logjoint, but transform back for the return traces.
"""
function hmc(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, eps::Float64; ad_backend::Symbol=:tracker)
    logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
    K = length(addresses_to_ix)
    samples = hmc_logjoint(logjoint, K, n_samples, L, eps, ad_backend=ad_backend)

    @assert samples isa AbstractMatrix
    retvals = Vector{Any}(undef, n_samples)
    @progress for i in axes(samples,2)
        X = view(samples,:,i)
        _, retvals[i] = transform_to_constrained!(X, model, args, observations, addresses_to_ix)
    end
    return StaticTraces(addresses_to_ix, samples, retvals)
end
# could be optimised by storing constrained values and return values in loop

export hmc