
const Param2Ix = Dict{Any, UnitRange{Int}}

mutable struct ParametersCollector <: StaticSampler
    params_to_ix::Param2Ix
    params_size::Int
    function ParametersCollector()
        return new(Param2Ix(),0)
    end
end

function sample(sampler::ParametersCollector, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    return rand(dist)
end

function param(sampler::ParametersCollector, addr::Any, size::Int=1)
    sampler.params_to_ix[addr] = (sampler.params_size+1):(sampler.params_size+size)
    sampler.params_size += size
    if size == 1
        return 0
    else
        return zeros(size)
    end
end

function get_params_to_ix(model::StaticModel, args::Tuple, observations::Dict)::Param2Ix
    sampler = ParametersCollector()
    model(args, sampler, observations)
    return sampler.params_to_ix
end

mutable struct GuideSampler{T,V} <: StaticSampler
    W::T
    params_to_ix::Param2Ix
    addresses_to_ix::Addr2Ix
    phi::V
    X::Vector{T}
    function GuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where {T <: Real, V <: AbstractVector{T}}
        return new{eltype(V),V}(0., params_to_ix, addresses_to_ix, phi, zeros(eltype(phi), length(addresses_to_ix)))
    end
end

function sample(sampler::GuideSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs) # TODO: assert no obs?
        return obs
    end
    value = rand(dist)
    sampler.X[sampler.addresses_to_ix[addr]] = value
    sampler.W += logpdf(dist, value)
    return value
end

function param(sampler::GuideSampler, addr::Any, size::Int=1)
    ix = sampler.params_to_ix[addr]
    if size == 1
        return sampler.phi[ix[1]]
    else
        return sampler.phi[ix]
    end
end

struct Guide <: VariationalDistribution
    sampler::GuideSampler
    model::StaticModel
    args::Tuple
    observations::Dict
end

function make_guide(model::StaticModel, args::Tuple, observations::Dict, addresses_to_ix::Addr2Ix)
    params_to_ix = get_params_to_ix(model, args, observations)
    N = sum(length(ix) for (_, ix) in params_to_ix)
    sampler = GuideSampler(params_to_ix, addresses_to_ix, zeros(N))
    return Guide(sampler, model, args, observations)
end

function initial_params(guide::Guide)::AbstractVector{<:Float64}
    nparams = sum(length(ix) for (_, ix) in guide.sampler.params_to_ix)
    return zeros(nparams)
end

function update_params(guide::Guide, params::AbstractVector{<:Float64})::VariationalDistribution
    new_sampler = GuideSampler(guide.sampler.params_to_ix, guide.sampler.addresses_to_ix, params)
    return Guide(new_sampler, guide.model, guide.args, guide.observations)
end

function rand_and_logpdf(guide::Guide)
    guide.sampler.W = 0.0
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X, guide.sampler.W
end

export make_guide