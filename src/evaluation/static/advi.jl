import TinyPPL.Distributions: mean, MeanFieldGaussian, FullRankGaussian, ELBOEstimator

struct StaticVIResult <: VIResult
    Q::VariationalDistribution
    addresses_to_ix::Addr2Ix
    transform_to_constrained!::Function # transform AbstractStaticTrace inplace and returns model retval
end
Base.show(io::IO, viresult::StaticVIResult) = print(io, "StaticVIResult($(viresult.Q))")

function sample_posterior(res::StaticVIResult, n::Int)
    samples = rand(res.Q, n)
    @assert samples isa AbstractMatrix
    retvals = Vector{Any}(undef, n)
    @progress for i in axes(samples,2)
        X = view(samples,:,i)
        _, retvals[i] = res.transform_to_constrained!(X)
    end
    return StaticTraces(res.addresses_to_ix, samples, retvals)
end
export sample_posterior

"""
Key idea: transform the static model to a logjoint function and use ADVI for logjoints.
See logjoint/ADVI.jl
"""

import TinyPPL.Logjoint: advi_meanfield_logjoint

function advi_meanfield(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
    K = length(addresses_to_ix)
    result = advi_meanfield_logjoint(logjoint, K, n_samples, L, learning_rate)
    _transform_to_constrained!(X::AbstractStaticTrace) = transform_to_constrained!(X, model, args, observations, addresses_to_ix)
    return StaticVIResult(result, addresses_to_ix, _transform_to_constrained!)
end

import TinyPPL.Logjoint: advi_fullrank_logjoint

function advi_fullrank(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
    K = length(addresses_to_ix)
    result = advi_fullrank_logjoint(logjoint, K, n_samples, L, learning_rate)
    _transform_to_constrained!(X::AbstractStaticTrace) = transform_to_constrained!(X, model, args, observations, addresses_to_ix)
    return StaticVIResult(result, addresses_to_ix, _transform_to_constrained!)
end

import TinyPPL.Logjoint: advi_logjoint

"""
ADVI with variational distributions given `q`.
Variational distribution is fitted by default to original model, but can also be fitted to unconstrained model,
by setting `unconstrained = true`.
ELBO is optimised with automatic differentiation (AD).
"""
function advi(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64,
    q::VariationalDistribution, estimator::ELBOEstimator;
    unconstrained::Bool=false, ad_backend::Symbol=:tracker)

    if unconstrained
        logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
        _transform_to_constrained(X::AbstractUniversalTrace) = transform_to_constrained(X, model, args, observations)
        _viresult_map! = _transform_to_constrained
    else
        logjoint, addresses_to_ix = make_logjoint(model, args, observations)
        _no_transform(X::AbstractUniversalTrace) = X, model(args, TraceSampler(X) , observations)
        _viresult_map! = _no_transform
    end

    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator, ad_backend=ad_backend)
    return StaticVIResult(result, addresses_to_ix, _viresult_map!)
end

"""
ADVI with variational distributions given by guide program.
Guide has to provide values in the correct support (absolute continuity).
Guide is fitted by default to original model, but can also be fitted to unconstrained model,
by setting `unconstrained = true`.
ELBO is optimised with automatic differentiation (AD).
"""
function advi(model::StaticModel, args::Tuple, observations::Observations,
    n_samples::Int, L::Int, learning_rate::Float64,
    guide::StaticModel, guide_args::Tuple, estimator::ELBOEstimator;
    unconstrained::Bool=false, ad_backend::Symbol=:tracker)

    if unconstrained
        logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
        _transform_to_constrained(X::AbstractUniversalTrace) = transform_to_constrained(X, model, args, observations)
        _viresult_map! = _transform_to_constrained
    else
        logjoint, addresses_to_ix = make_logjoint(model, args, observations)
        _no_transform(X::AbstractUniversalTrace) = X, model(args, TraceSampler(X) , observations)
        _viresult_map! = _no_transform
    end

    q = make_guide(guide, guide_args, addresses_to_ix)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator, ad_backend=ad_backend)
    return StaticVIResult(result, addresses_to_ix, _viresult_map!)
end

export advi_meanfield, advi_fullrank, advi