
mutable struct LogJoint{T} <: StaticSampler
    W::Float64
    addresses_to_ix::Addr2Ix
    X::T
    function LogJoint(addresses_to_ix::Addr2Ix, X::T) where T <: AbstractVector{<:Real}
        return new{T}(0., addresses_to_ix, X)
    end
end

function sample(sampler::LogJoint, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
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
    function logjoint(X::AbstractVector{<:Real})
        sampler = LogJoint(addresses_to_ix, X)
        model(args, sampler, observations)
        return sampler.W
    end
    return addresses_to_ix, logjoint
end