
import TinyPPL.Distributions: MeanField, init_variational_distribution, mean, mode
import TinyPPL.Distributions: ReinforceELBO

"""
Determine variational distribution for each address.
Assumes that distribution type is static for each address,
"""
struct MeanFieldCollector <: StaticSampler
    dists::Vector{VariationalDistribution}
    addresses_to_ix::Addr2Ix
end

function sample(sampler::MeanFieldCollector, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    value = dist isa Distributions.DiscreteDistribution ? mode(dist) : mean(dist)
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

"""
BBVI, where we fit unconstrained model.
VariationalDistributions are automatically determined with MeanFieldCollector, which uses init_variational_distribution.
This method *uses* AD to compute gradient of REINFORCE approximation.
"""
function bbvi(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint, addresses_to_ix = make_unconstrained_logjoint(model, args, observations)
    q = get_mixed_meanfield(model, args, observations, addresses_to_ix)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, ReinforceELBO())
    _transform_to_constrained!(X::AbstractStaticTrace) = transform_to_constrained!(X, model, args, observations, addresses_to_ix)
    return StaticVIResult(result, addresses_to_ix, transform_to_constrained!)
end
export bbvi