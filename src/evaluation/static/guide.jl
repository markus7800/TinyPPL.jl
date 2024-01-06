import Distributions
import TinyPPL.Distributions: mean, mode

mutable struct ParametersCollector <: StaticSampler
    params_to_ix::Param2Ix
    params_size::Int
    function ParametersCollector()
        return new(Param2Ix(),0)
    end
end

function sample(sampler::ParametersCollector, addr::Address, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    return  dist isa Distribution.DiscreteDistribution ? mode(dist) : mean(dist)
end

function param(sampler::ParametersCollector, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())
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

mutable struct StaticGuideSampler{T,V} <: StaticSampler
    W::T # depends on the eltype of phi
    params_to_ix::Param2Ix
    addresses_to_ix::Addr2Ix
    phi::V
    X::Vector{T}
    function StaticGuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where {T <: Real, V <: AbstractVector{T}}
        return new{eltype(V),V}(0., params_to_ix, addresses_to_ix, phi, zeros(eltype(phi), length(addresses_to_ix)))
    end
end

function sample(sampler::StaticGuideSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs) # TODO: assert no obs?
        return obs
    end
    value = rand(dist)
    sampler.X[sampler.addresses_to_ix[addr]] = value
    sampler.W += logpdf(dist, value)
    return value
end

function param(sampler::StaticGuideSampler, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())
    ix = sampler.params_to_ix[addr]
    if size == 1
        return constrain_param(constraint, sampler.phi[ix[1]])
    else
        return constrain_param(constraint, sampler.phi[ix])
    end
end

struct StaticGuide <: VariationalDistribution
    sampler::StaticGuideSampler
    model::StaticModel
    args::Tuple
    observations::Dict
end

# guide can be used for unconstrained and constrained logjoint
function make_guide(model::StaticModel, args::Tuple, observations::Dict, addresses_to_ix::Addr2Ix)
    params_to_ix = get_params_to_ix(model, args, observations)
    N = sum(length(ix) for (_, ix) in params_to_ix)
    sampler = StaticGuideSampler(params_to_ix, addresses_to_ix, zeros(N))
    return StaticGuide(sampler, model, args, observations)
end

# import TinyPPL.Distributions: initial_params
# function initial_params(guide::Guide)::AbstractVector{<:Float64}
#     nparams = sum(length(ix) for (_, ix) in guide.sampler.params_to_ix)
#     return zeros(nparams)
# end

import TinyPPL.Distributions: get_params
function get_params(q::StaticGuide)::AbstractVector{<:Real}
    return q.sampler.phi
end

import TinyPPL.Distributions: update_params
function update_params(guide::StaticGuide, params::AbstractVector{<:Float64})::VariationalDistribution
    # since StaticGuideSampler is generic type, we freshly instantiate
    new_sampler = StaticGuideSampler(guide.sampler.params_to_ix, guide.sampler.addresses_to_ix, params)
    return StaticGuide(new_sampler, guide.model, guide.args, guide.observations)
end

import TinyPPL.Distributions: rand_and_logpdf
function rand_and_logpdf(guide::StaticGuide)
    guide.sampler.W = 0.0
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X, guide.sampler.W
end

function Distributions.rand(guide::StaticGuide)
    guide.sampler.W = 0.0
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X
end

# TODO?
# function Distributions.logpdf(guide::StaticGuide, X)
#     guide.sampler.W = 0.0
#     guide.sampler.X = X
#     guide.model(guide.args, guide.sampler, guide.observations)
#     return guide.sampler.W
# end

function Distributions.rand(guide::StaticGuide, n::Int)
    return reduce(hcat, Distribution.rand(guide) for _ in 1:n)
end


export make_guide

mutable struct StaticParameterTransformer <: StaticSampler
    phi::VariationalParameters
    params_to_ix::Param2Ix
    transformed_phi::VariationalParameters
    function StaticParameterTransformer(guide::StaticGuide)
        return new(guide.sampler.phi, guide.sampler.params_to_ix, similar(guide.sampler.phi))
    end
end

function sample(sampler::StaticParameterTransformer, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    return dist isa Distribution.DiscreteDistribution ? mode(dist) : mean(dist)
end

function param(sampler::StaticParameterTransformer, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())
    ix = sampler.params_to_ix[addr]
    if size == 1
        parameters = constrain_param(constraint, sampler.phi[ix[1]])
    else
        parameters = constrain_param(constraint, sampler.phi[ix])
    end
    sampler.transformed_phi[ix] .= parameters
    return parameters
end


# struct VIParameters
#     phi::VariationalParameters
#     params_to_ix::Param2Ix
# end
# function Base.show(io::IO, p::VIParameters)
#     print(io, "VIParameters(")
#     print(io, sort(collect(keys(p.params_to_ix)), lt = (x,y) -> first(p.params_to_ix[x]) < first(p.params_to_ix[y])))
#     print(io, ")")
# end
# function Base.getindex(p::VIParameters, i::Int)
#     return p.phi[i]
# end
# function Base.getindex(p::VIParameters, addr::Any)
#     ix = p.params_to_ix[addr]
#     if length(ix) == 1
#         return p.phi[ix[1]]
#     else
#         return p.phi[ix]
#     end
# end

"""
Extracts parameters of guide and transforms them to the specified constraints.
"""
function get_constrained_parameters(guide::StaticGuide)
    sampler = StaticParameterTransformer(guide)
    guide.model(guide.args, sampler, guide.observations)
    return VIParameters(sampler.transformed_phi, guide.sampler.params_to_ix)
end

export get_constrained_parameters