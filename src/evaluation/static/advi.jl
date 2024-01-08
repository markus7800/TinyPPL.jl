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

function advi(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    _transform_to_constrained!(X::AbstractStaticTrace) = transform_to_constrained!(X, model, args, observations, addresses_to_ix)
    return StaticVIResult(result, addresses_to_ix, _transform_to_constrained!)
end

function advi(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64, guide::StaticModel, guide_args::Tuple, estimator::ELBOEstimator)
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)
    q = make_guide(guide, guide_args, Dict(), addresses_to_ix)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    # guide has to propose in correct support
    # if you want to fit guide to unconstrained model you have to do it manually and transform to constrained
    return StaticVIResult(result, addresses_to_ix, identity)
end

export advi_meanfield, advi_fullrank, advi

function get_number_of_latent_variables(model::StaticModel, args::Tuple, observations::Observations)
    return length(get_addresses(model, args, observations))
end
export get_number_of_latent_variables
