
import TinyPPL.Distributions: MeanField, init_variational_distribution, mean, mode

struct MeanFieldCollector <: StaticSampler
    dists::Vector{VariationalDistribution}
    addresses_to_ix::Addr2Ix
end

function sample(sampler::MeanFieldCollector, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    value = dist isa Distribution.DiscreteDistribution ? mode(dist) : mean(dist)
    ix = sampler.addresses_to_ix[addr]
    sampler.dists[ix] = init_variational_distribution(dist)
    return value
end
function get_mixed_meanfield(model::StaticModel, args::Tuple, observations::Observations, addresses_to_ix::Addr2Ix)::MeanField
    sampler = MeanFieldCollector(Vector{VariationalDistribution}(undef, length(addresses_to_ix)), addresses_to_ix)
    model(args, sampler, observations)
    return MeanField(sampler.dists)
end
export get_mixed_meanfield

function bbvi(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    q = get_mixed_meanfield(model, args, observations, ulj.addresses_to_ix)
    result = advi_logjoint(ulj.logjoint, n_samples, L, learning_rate, q, ReinforceELBO())
    return StaticVIResult(result, ulj.addresses_to_ix, ulj.transform_to_constrained!)
end
export bbvi