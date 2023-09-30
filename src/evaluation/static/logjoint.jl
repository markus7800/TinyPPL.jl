

mutable struct LogJoint <: StaticSampler
    W::Float64
    addresses_to_ix::Addr2Ix
    X::Vector{Real}
    function LogJoint(addresses_to_ix::Addr2Ix)
        return new(0., addresses_to_ix, Vector{Real}())
    end
end

function sample(sampler::StaticSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    value = sampler.X[sampler.addresses_to_ix[addr]]
    sampler.W += logpdf(dist, value)
    return value
end

function make_logjoint(model::StaticModel, args::Tuple, observations::Dict)
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(addresses)
    sampler = LogJoint(addresses_to_ix)
    return function logjoint(X::Vector{Real})
        sampler.X = X
        model(args, sampler, observations)
        return sampler.W
    end
end