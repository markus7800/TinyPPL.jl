import TinyPPL.Logjoint: advi_meanfield_logjoint

function advi_meanfield(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    addresses_to_ix, logjoint, transform_to_constrained!, _ = make_unconstrained_logjoint(model, args, observations)
    K = length(addresses_to_ix)
    result = advi_meanfield_logjoint(logjoint, K, n_samples, L, learning_rate)
    return result, transform_to_constrained!
end

import TinyPPL.Logjoint: advi_fullrank_logjoint

function advi_fullrank(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64)
    addresses_to_ix, logjoint, transform_to_constrained!, _ = make_unconstrained_logjoint(model, args, observations)
    K = length(addresses_to_ix)
    result = advi_fullrank_logjoint(logjoint, K, n_samples, L, learning_rate)
    return result, transform_to_constrained!
end

import TinyPPL.Logjoint: advi_logjoint, ELBOEstimator, MonteCarloELBO, RelativeEntropyELBO 

function advi(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    addresses_to_ix, logjoint, transform_to_constrained!, _ = make_unconstrained_logjoint(model, args, observations)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    return result, transform_to_constrained!
end

export advi_meanfield, advi_fullrank, advi_logjoint

import ..Distributions: MixedMeanField, init_variational_distribution

struct MixedMeanFieldCollector <: StaticSampler
    dists::Vector{VariationalDistribution}
    addresses_to_ix::Addr2Ix
end
function sample(sampler::MixedMeanFieldCollector, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    value = rand(dist)
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