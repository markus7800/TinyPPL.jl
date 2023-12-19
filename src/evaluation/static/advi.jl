import TinyPPL.Logjoint: advi_meanfield_logjoint
import ..Distributions: mean

function advi_meanfield(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    K = length(ulj.addresses_to_ix)
    result = advi_meanfield_logjoint(ulj.logjoint, K, n_samples, L, learning_rate)
    return result, ulj # TODO: maybe variational return type
end

import TinyPPL.Logjoint: advi_fullrank_logjoint

function advi_fullrank(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    K = length(ulj.addresses_to_ix)
    result = advi_fullrank_logjoint(ulj.logjoint, K, n_samples, L, learning_rate)
    return result, ulj
end

import TinyPPL.Logjoint: advi_logjoint
import TinyPPL.Distributions: ELBOEstimator, ReinforceELBO

function advi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    ulj = make_unconstrained_logjoint(model, args, observations)
    result = advi_logjoint(ulj.logjoint, n_samples, L, learning_rate, q, estimator)
    return result, ulj
end

function advi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64, guide::StaticModel, guide_args::Tuple, estimator::ELBOEstimator)
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)
    q = make_guide(guide, guide_args, Dict(), addresses_to_ix)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    return result, (logjoint, addresses_to_ix)
end

export advi_meanfield, advi_fullrank, advi

import ..Distributions: MixedMeanField, init_variational_distribution

struct MixedMeanFieldCollector <: StaticSampler
    dists::Vector{VariationalDistribution}
    addresses_to_ix::Addr2Ix
end
function sample(sampler::MixedMeanFieldCollector, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    value = mean(dist)
    ix = sampler.addresses_to_ix[addr]
    sampler.dists[ix] = init_variational_distribution(dist)
    return value
end
function get_mixed_meanfield(model::StaticModel, args::Tuple, observations::Dict, addresses_to_ix::Addr2Ix)::MixedMeanField
    sampler = MixedMeanFieldCollector(Vector{VariationalDistribution}(undef, length(addresses_to_ix)), addresses_to_ix)
    model(args, sampler, observations)
    return MixedMeanField(sampler.dists)
end
export get_mixed_meanfield

function bbvi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    q = get_mixed_meanfield(model, args, observations, ulj.addresses_to_ix)
    result = advi_logjoint(ulj.logjoint, n_samples, L, learning_rate, q, ReinforceELBO())
    return result, ulj
end
export bbvi

function get_number_of_latent_variables(model::StaticModel, args::Tuple, observations::Dict)
    return length(get_addresses(model, args, observations))
end
export get_number_of_latent_variables
