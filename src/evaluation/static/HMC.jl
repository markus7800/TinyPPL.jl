import TinyPPL.Logjoint: hmc_logjoint

"""
The hmc methods for static models also calls the logjoint implementation.
If `unconstrained=true`, we sample the unconstrained logjoint, but transform back for the return traces.
`x_initial` is a trace with constrained values (it is not mutated).
If no `x_initial` is given, the chain starts at a sample from the prior (which lies in the support).
"""
function hmc(model::StaticModel, args::Tuple, observations::AbstractObservations,
    n_samples::Int, L::Int, eps::Float64;
    ad_backend::Symbol=:tracker, x_initial::Union{Nothing,Vector{Float64}}=nothing, unconstrained::Bool=false)

    if unconstrained
        logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
    else
        logjoint, addresses_to_ix = make_logjoint(model, args, observations)
    end
    K = length(addresses_to_ix)

    if isnothing(x_initial)
        x_initial = Vector{Float64}(sample_from_prior(model, args, observations, 1).data[:,1])
    else
        x_initial = copy(x_initial)
    end
    if unconstrained
        transform_to_unconstrained!(x_initial, model, args, observations, addresses_to_ix)
    end

    samples = hmc_logjoint(logjoint, K, n_samples, L, eps, ad_backend=ad_backend, x_initial=x_initial)

    @assert samples isa AbstractMatrix
    retvals = Vector{Any}(undef, n_samples)
    @progress for i in axes(samples,2)
        X = view(samples,:,i)
        if unconstrained
            _, retvals[i] = transform_to_constrained!(X, model, args, observations, addresses_to_ix)
        else
            retvals[i] = model(args, StaticForward(addresses_to_ix, X), observations)
        end
    end
    return StaticTraces(addresses_to_ix, samples, retvals)
end
# could be optimised by storing constrained values and return values in loop

export hmc