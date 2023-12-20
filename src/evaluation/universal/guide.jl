import Tracker

const Param2Ix = Dict{Any, UnitRange{Int}}

mutable struct GuideSampler{T,V} <: UniversalSampler
    W::T # depends on the eltype of phi
    params_to_ix::Param2Ix
    phi::V # Vector{Float64} or TrackedVector
    X::Dict{Any,Real}
    constraints::Dict{Any,ParamConstraint}
    function GuideSampler(params_to_ix::Param2Ix, phi::V, constraints=Dict{Any,ParamConstraint}()) where {T <: Real, V <: AbstractVector{T}}
        return new{eltype(V),V}(0., params_to_ix, phi, Dict{Any,Real}(), constraints)
    end
end

function sample(sampler::GuideSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs) # TODO: assert no obs?
        return obs
    end
    value = get!(sampler.X, addr, rand(dist))
    sampler.W += logpdf(dist, value)
    return value
end

function param(sampler::GuideSampler, addr::Any, size::Int=1, constraint::ParamConstraint=Unconstrained())
    if !haskey(sampler.params_to_ix, addr)
        n = length(sampler.phi)
        ix = (n+1):(n+size)
        sampler.params_to_ix[addr] = ix
        sampler.constraints[addr] = constraint
        # all parameters are initialised to 0
        if Tracker.istracked(sampler.phi)
            sampler.phi = vcat(sampler.phi, Tracker.param(zeros(size)))
        else
            sampler.phi = vcat(sampler.phi, zeros(eltype(sampler.phi), size))
        end
    end
    ix = sampler.params_to_ix[addr]
    if size == 1
        return transform(constraint, sampler.phi[ix[1]])
    else
        return transform(constraint, sampler.phi[ix])
    end
end

import TinyPPL.Distributions: VariationalDistribution

struct Guide <: VariationalDistribution
    sampler::GuideSampler
    model::UniversalModel
    args::Tuple
    observations::Dict
end

# guide can be used for unconstrained and constrained logjoint
function make_guide(model::UniversalModel, args::Tuple, observations::Dict)::Guide
    sampler = GuideSampler(Param2Ix(), zeros(0))
    return Guide(sampler, model, args, observations)
end
export make_guide

# import ..Distributions: initial_params
# function initial_params(guide::Guide)::AbstractVector{<:Float64}
#     nparams = sum(length(ix) for (_, ix) in guide.sampler.params_to_ix)
#     return zeros(nparams)
# end

import ..Distributions: get_params
function get_params(q::Guide)::AbstractVector{<:Real}
    return q.sampler.phi
end

import ..Distributions: update_params
function update_params(guide::Guide, params::AbstractVector{<:Float64})::VariationalDistribution
    # since GuideSampler is generic type, we freshly instantiate
    new_sampler = GuideSampler(guide.sampler.params_to_ix, params, guide.sampler.constraints)
    return Guide(new_sampler, guide.model, guide.args, guide.observations)
end

import ..Distributions: rand_and_logpdf
function rand_and_logpdf(guide::Guide)
    guide.sampler.W = 0.0
    guide.sampler.X = Dict{Any,Real}()
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X, guide.sampler.W
end

import Distributions
function Distributions.rand(guide::Guide)
    guide.sampler.W = 0.0
    guide.sampler.X = Dict{Any,Real}()
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X
end

function Distributions.logpdf(guide::Guide, X::Dict{Any,Real})
    guide.sampler.W = 0.0
    guide.sampler.X = X
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.W
end

function Distributions.rand(guide::Guide, n::Int)
    return [Distribution.rand(guide) for _ in 1:n]
end


struct Parameters
    phi::AbstractVector{<:Real}
    params_to_ix::Param2Ix
end
function Base.show(io::IO, p::Parameters)
    print(io, "Parameters(")
    print(io, p.phi)
    print(io, ")")
end
function Base.getindex(p::Parameters, i::Int)
    return p.phi[i]
end
# function Base.getindex(p::Parameters, addrs::AbstractVector{<:Any})
#     return [p.phi[p.params_to_ix[addr]] for addr in addrs]
# end 
function Base.getindex(p::Parameters, addr::Any)
    ix = p.params_to_ix[addr]
    if length(ix) == 1
        return p.phi[ix[1]]
    else
        return p.phi[ix]
    end
end 

function get_constrained_parameters(guide::Guide)
    transformed_phi = similar(guide.sampler.phi)
    for (addr, ix) in guide.sampler.params_to_ix
        transformed_phi[ix] = transform(guide.sampler.constraints[addr], guide.sampler.phi[ix])
    end
    return Parameters(transformed_phi, guide.sampler.params_to_ix)
end

export get_constrained_parameters