import Tracker
import TinyPPL.Distributions: VariationalParameters

"""
Samples values or reuses values of trace `X` and computes log Q(X).
Parameter name are directly mapped to their values in `phi`.
If we encounter parameter with unknown address, then we add the new parameter to `phi`.
Parameters are always initiliased to 0.
Also, the parameter constraints are recorded which are assumed to be *static*.
"""
mutable struct UniversalGuideSampler{V} <: UniversalSampler
    W::Real                                 # log Q(X), type depends on the eltype of phi and X
    phi::Dict{Address,V}                    # mapping of params to address
    X::UniversalTrace                       # Sample trace of guide, values will be sampled if not provided 
    constraints::Dict{Address,ParamConstraint}
    function UniversalGuideSampler(phi::Dict{Address,V}, constraints=Dict{Address,ParamConstraint}()) where {V <: VariationalParameters}
        return new{V}(0., phi, UniversalTrace(), constraints)
    end
end

function sample(sampler::UniversalGuideSampler, addr::Address, dist::Distribution, obs::RVValue)::RVValue
    error("A guide program should not have observed values")
end

function sample(sampler::UniversalGuideSampler, addr::Address, dist::Distribution, obs::Nothing)::RVValue
    # if there is a value in trace use it else sample
    if !haskey(sampler.X, addr)
        sampler.X[addr] = rand(dist)
    end
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    # println("guide: ", addr, " ", value, " ", dist)
    return value
end

function param(sampler::UniversalGuideSampler{V}, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained()) where V <: VariationalParameters
    if !haskey(sampler.phi, addr)
        # add new parameter
        sampler.constraints[addr] = constraint
        # all parameters are initialised to 0, vcat to preserve TrackedVector type
        if V == Vector{Float64}
            # sampler.phi = vcat(sampler.phi, zeros(eltype(sampler.phi), size))
            sampler.phi[addr] = zeros(size)
        else
            # This does not feed back grads correctly, because phi is not leaf?
            # sampler.phi = vcat(sampler.phi, Tracker.param(zeros(size)))
            sampler.phi[addr] = Tracker.param(zeros(size))
        end
    end
    # inject parameter
    if size == 1
        return constrain_param(constraint, sampler.phi[addr][1])
    else
        return constrain_param(constraint, sampler.phi[addr])
    end
end

import TinyPPL.Distributions: VariationalDistribution

"""
Wraps `UniversalGuideSampler` and guide program.
Implements similar interface to VariationalDistribution.
"""
struct UniversalGuide
    sampler::UniversalGuideSampler
    model::UniversalModel
    args::Tuple
    observations::Observations
end

# guide can be used for unconstrained and constrained logjoint
function make_guide(model::UniversalModel, args::Tuple)::UniversalGuide
    sampler = UniversalGuideSampler(Dict{Address,Vector{Float64}}())
    return UniversalGuide(sampler, model, args, Observations())
end
export make_guide

# deliberate violation of VariationalDistribution interface types
import TinyPPL.Distributions: get_params
function get_params(q::UniversalGuide)::Dict{Address,<:VariationalParameters}
    return q.sampler.phi
end

# deliberate violation of VariationalDistribution interface types
import TinyPPL.Distributions: update_params
function update_params(guide::UniversalGuide, params::Dict{Address,<:VariationalParameters})::VariationalDistribution
    new_sampler = UniversalGuideSampler(params, copy(guide.sampler.constraints))
    return UniversalGuide(new_sampler, guide.model, guide.args, guide.observations)
end

import TinyPPL.Distributions: rand_and_logpdf
function rand_and_logpdf(guide::UniversalGuide)
    guide.sampler.W = 0.0
    guide.sampler.X = UniversalTrace()
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X, guide.sampler.W
end

import Distributions
function Distributions.rand(guide::UniversalGuide)
    guide.sampler.W = 0.0
    guide.sampler.X = UniversalTrace()
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.X
end

# This is wasteful because we have to rerun model.
# If possible use rand_and_logpdf instead.
function Distributions.logpdf(guide::UniversalGuide, X::UniversalTrace)
    guide.sampler.W = 0.0
    guide.sampler.X = X
    guide.model(guide.args, guide.sampler, guide.observations)
    return guide.sampler.W
end

function Distributions.rand(guide::UniversalGuide, n::Int)
    return [Distributions.rand(guide) for _ in 1:n]
end

struct UniversalVIParameters <: VIParameters
    phi::Dict{Address, <:VariationalParameters}
end
function Base.show(io::IO, p::UniversalVIParameters)
    print(io, "UniversalVIParameters(")
    print(io, sort(collect(keys(p.phi))))
    print(io, ")")
end
function Base.getindex(p::UniversalVIParameters, addr::Address)
    params = p.phi[addr]
    if length(params) == 1
        return params[1]
    else
        return params
    end
end

"""
Extracts parameters of guide and transforms them to the specified *static* constraints.
UniversalParameterTransformer with dynamic constraints does not work,
since we do not know if all parameters are encountered in model execution.
"""
function get_constrained_parameters(guide::UniversalGuide)
    transformed_phi = empty(guide.sampler.phi)
    for (addr, params) in guide.sampler.phi
        transformed_phi[addr] = constrain_param(guide.sampler.constraints[addr], params)
    end
    return UniversalVIParameters(transformed_phi)
end

export get_constrained_parameters