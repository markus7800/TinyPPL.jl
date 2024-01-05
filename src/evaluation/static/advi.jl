import TinyPPL.Logjoint: advi_meanfield_logjoint
import TinyPPL.Distributions: mean, MeanFieldGaussian, FullRankGaussian

struct StaticVIResult
    Q::VariationalDistribution
    addresses_to_ix::Addr2Ix
    transform_to_constrained!::Function
end

function sample_posterior(res::StaticVIResult, n::Int)
    samples = rand(res.Q, n)
    @assert samples isa AbstractMatrix
    res.transform_to_constrained!(samples)
    return Traces(res.addresses_to_ix, samples)
end
export sample_posterior

function advi_meanfield(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    K = length(ulj.addresses_to_ix)
    result = advi_meanfield_logjoint(ulj.logjoint, K, n_samples, L, learning_rate)
    return StaticVIResult(result, ulj.addresses_to_ix, ulj.transform_to_constrained!)
end

import TinyPPL.Logjoint: advi_fullrank_logjoint

function advi_fullrank(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    K = length(ulj.addresses_to_ix)
    result = advi_fullrank_logjoint(ulj.logjoint, K, n_samples, L, learning_rate)
    return StaticVIResult(result, ulj.addresses_to_ix, ulj.transform_to_constrained!)
end

import TinyPPL.Logjoint: advi_logjoint
import TinyPPL.Distributions: ELBOEstimator, ReinforceELBO

function advi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    ulj = make_unconstrained_logjoint(model, args, observations)
    result = advi_logjoint(ulj.logjoint, n_samples, L, learning_rate, q, estimator)
    return StaticVIResult(result, ulj.addresses_to_ix, ulj.transform_to_constrained!)
end

function advi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64, guide::StaticModel, guide_args::Tuple, estimator::ELBOEstimator)
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)
    q = make_guide(guide, guide_args, Dict(), addresses_to_ix)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    # guide has to propose in correct support
    # if you want to fit guide to unconstrained model you have to do it manually and transform to constrained
    return StaticVIResult(result, addresses_to_ix, identity)
end

export advi_meanfield, advi_fullrank, advi

import TinyPPL.Distributions: MeanField, init_variational_distribution

struct MeanFieldCollector <: StaticSampler
    dists::Vector{VariationalDistribution}
    addresses_to_ix::Addr2Ix
end
function sample(sampler::MeanFieldCollector, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    value = mean(dist)
    ix = sampler.addresses_to_ix[addr]
    sampler.dists[ix] = init_variational_distribution(dist)
    return value
end
function get_mixed_meanfield(model::StaticModel, args::Tuple, observations::Dict, addresses_to_ix::Addr2Ix)::MeanField
    sampler = MeanFieldCollector(Vector{VariationalDistribution}(undef, length(addresses_to_ix)), addresses_to_ix)
    model(args, sampler, observations)
    return MeanField(sampler.dists)
end
export get_mixed_meanfield

function bbvi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    q = get_mixed_meanfield(model, args, observations, ulj.addresses_to_ix)
    result = advi_logjoint(ulj.logjoint, n_samples, L, learning_rate, q, ReinforceELBO())
    return StaticVIResult(result, ulj.addresses_to_ix, ulj.transform_to_constrained!)
end
export bbvi

function get_number_of_latent_variables(model::StaticModel, args::Tuple, observations::Dict)
    return length(get_addresses(model, args, observations))
end
export get_number_of_latent_variables
