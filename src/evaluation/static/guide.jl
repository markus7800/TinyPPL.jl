import Distributions
import TinyPPL.Distributions: mean, mode, VariationalParameters, VariationalDistribution

const Param2Ix = Dict{Address, UnitRange{Int}}

"""
Collects all parameters in a model by executing it.
Maps parameters to index in vector.
Names are mapped to ranges i:j, where j-i+1 is the size of parameter vector for name.
"""
mutable struct ParametersCollector <: StaticSampler
    params_to_ix::Param2Ix
    params_size::Int
    function ParametersCollector()
        return new(Param2Ix(),0)
    end
end

function sample(sampler::ParametersCollector, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    return  dist isa Distributions.DiscreteDistribution ? mode(dist) : mean(dist)
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

import Tracker
import ForwardDiff
import ReverseDiff

"""
For given parameteers `phi` this sampler generates a trace into `X`.
Type-optimised for generating (tracked) samples.
Cannot evaluate logjoint in contrast to UniversalGuideSampler.
"""
mutable struct StaticGuideSampler{T,V} <: StaticSampler
    W::T # depends on the eltype of phi
    params_to_ix::Param2Ix
    addresses_to_ix::Addr2Ix
    phi::V
    X::Vector{T}
    function StaticGuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where V <: Vector{Float64}
        return new{Float64,V}(0., params_to_ix, addresses_to_ix, phi, zeros(Float64,length(addresses_to_ix)))
    end

    function StaticGuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where V <: Tracker.TrackedVector{Float64,Vector{Float64}}
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, zeros(Tracker.TrackedReal{Float64}, length(addresses_to_ix)))
    end
    # function StaticGuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where V <: Vector{Tracker.TrackedReal{Float64}}
    #     return new{Tracker.TrackedReal{Float64},V}(0., params_to_ix, addresses_to_ix, phi, zeros(Tracker.TrackedReal{Float64}, length(addresses_to_ix)))
    # end

    function StaticGuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where V <: Vector{<:ForwardDiff.Dual}
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, zeros(Tracker.TrackedReal{Float64}, length(addresses_to_ix)))
    end

    function StaticGuideSampler(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V) where V <: ReverseDiff.TrackedArray
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, zeros(Tracker.TrackedReal{Float64}, length(addresses_to_ix)))
    end
end

function sample(sampler::StaticGuideSampler, addr::Address, dist::Distribution, obs::RVValue)::RVValue
    error("A guide program should not have observed values")
end

function sample(sampler::StaticGuideSampler, addr::Address, dist::Distribution, obs::Nothing)::RVValue
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

"""
Wraps `StaticGuideSampler` and guide program.
Implements the VariationalDistribution interface.
"""
struct StaticGuide <: VariationalDistribution
    sampler::StaticGuideSampler
    model::StaticModel
    args::Tuple
    observations::Observations
end
Base.show(io::IO, guide::StaticGuide) = print(io, "Guide($(guide.model.f))")

# guide can be used for unconstrained and constrained logjoint
function make_guide(model::StaticModel, args::Tuple, addresses_to_ix::Addr2Ix)
    params_to_ix = get_params_to_ix(model, args, Observations())
    N = sum(length(ix) for (_, ix) in params_to_ix)
    sampler = StaticGuideSampler(params_to_ix, addresses_to_ix, zeros(N))
    return StaticGuide(sampler, model, args, Observations())
end
export make_guide

import TinyPPL.Distributions: get_params
function get_params(q::StaticGuide)::AbstractVector{<:Real}
    return q.sampler.phi
end

import TinyPPL.Distributions: update_params
function update_params(guide::StaticGuide, params::VariationalParameters)::VariationalDistribution
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

function Distributions.rand(guide::StaticGuide, n::Int)
    K = length(guide.sampler.addresses_to_ix)
    X = Array{Real}(undef, K, n) # TODO: Real here is ugly
    for i in 1:N
        X[:,i] = Distributions.rand(guide)
    end
    return X
end

"""
For given parameteers `phi` this sampler scores a trace `X`.
"""
mutable struct StaticGuideScorer{T,V} <: StaticSampler
    W::T
    params_to_ix::Param2Ix
    addresses_to_ix::Addr2Ix
    phi::V
    X::AbstractStaticTrace
    function StaticGuideScorer(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V, X::Vector{<:Real}) where {V <: Vector{Float64}}
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, X)
    end
    
    function StaticGuideScorer(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V, X::Vector{<:Real}) where {V <: Tracker.TrackedVector}
        return new{Tracker.TrackedReal,V}(0., params_to_ix, addresses_to_ix, phi, X)
    end

    function StaticGuideScorer(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V, X::Vector{<:Real}) where {V <: Vector{<:ForwardDiff.Dual}}
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, X)
    end

    function StaticGuideScorer(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V, X::Vector{<:Real}) where {V <: ReverseDiff.TrackedArray}
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, X)
    end

    function StaticGuideScorer(params_to_ix::Param2Ix, addresses_to_ix::Addr2Ix, phi::V, X::Vector{<:ForwardDiff.Dual}) where {V <: Vector{<:ForwardDiff.Dual}}
        return new{Real,V}(0., params_to_ix, addresses_to_ix, phi, X)
    end
end

function sample(sampler::StaticGuideScorer, addr::Address, dist::Distribution, obs::RVValue)::RVValue
    error("A guide program should not have observed values")
end

function sample(sampler::StaticGuideScorer, addr::Address, dist::Distribution, obs::Nothing)::RVValue
    value = sampler.X[sampler.addresses_to_ix[addr]]
    sampler.W += logpdf(dist, value)
    return value
end

function param(sampler::StaticGuideScorer, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())
    ix = sampler.params_to_ix[addr]
    if size == 1
        return constrain_param(constraint, sampler.phi[ix[1]])
    else
        return constrain_param(constraint, sampler.phi[ix])
    end
end

function Distributions.logpdf(guide::StaticGuide, X::AbstractStaticTrace)
    scorer = StaticGuideScorer(guide.sampler.params_to_ix, guide.sampler.addresses_to_ix, guide.sampler.phi, X)
    guide.model(guide.args, scorer, guide.observations)
    return scorer.W
end

"""
For parameters `phi`, run the model and transform them accorinding to their static constraints.
Result is stored in `transformed_phi`.
"""
mutable struct StaticParameterTransformer <: StaticSampler
    phi::VariationalParameters
    params_to_ix::Param2Ix
    transformed_phi::VariationalParameters
    function StaticParameterTransformer(guide::StaticGuide)
        return new(guide.sampler.phi, guide.sampler.params_to_ix, similar(guide.sampler.phi))
    end
end

function sample(sampler::StaticParameterTransformer, addr::Address, dist::Distribution, obs::Nothing)::RVValue
    return dist isa Distributions.DiscreteDistribution ? mode(dist) : mean(dist)
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


struct StaticVIParameters <: VIParameters
    phi::VariationalParameters
    params_to_ix::Param2Ix
end
function Base.show(io::IO, p::StaticVIParameters)
    print(io, "StaticVIParameters(")
    print(io, sort(collect(keys(p.params_to_ix)), lt = (x,y) -> first(p.params_to_ix[x]) < first(p.params_to_ix[y])))
    print(io, ")")
end
function Base.getindex(p::StaticVIParameters, addr::Address)
    ix = p.params_to_ix[addr]
    if length(ix) == 1
        return p.phi[ix[1]]
    else
        return p.phi[ix]
    end
end

"""
Extracts parameters of guide and transforms them to the specified constraints.
"""
function get_constrained_parameters(guide::StaticGuide)
    sampler = StaticParameterTransformer(guide)
    guide.model(guide.args, sampler, guide.observations)
    return StaticVIParameters(sampler.transformed_phi, guide.sampler.params_to_ix)
end

export get_constrained_parameters