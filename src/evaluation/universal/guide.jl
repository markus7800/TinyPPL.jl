import Tracker
import TinyPPL.Distributions: VariationalParameters

"""
Samples values or reuses values of trace `X` and computes log Q(X).
Parameters are used by indexing into `phi` with `params_to_ix`.
If we encounter parameter with unknown address, then we expand `phi` with the new parameter.
Parameters are always initiliased to 0.
Also, the parameter constraints are recorded which are assumed to be *static*.
"""
mutable struct UniversalGuideSampler{V} <: UniversalSampler
    W::Real                                 # log Q(X), type depends on the eltype of phi and X
    params_to_ix::Param2Ix                  # mapping of parameter address to index in phi
    phi::V                                  # vector of parameters Vector{Float64} or TrackedVector
    X::UniversalTrace                       # Sample trace of guide, values will be sampled if not provided 
    constraints::Dict{Any,ParamConstraint}
    function UniversalGuideSampler(params_to_ix::Param2Ix, phi::V, constraints=Dict{Address,ParamConstraint}()) where {V <: VariationalParameters}
        return new{V}(0., params_to_ix, phi, UniversalTrace(), constraints)
    end
end

function sample(sampler::UniversalGuideSampler, addr::Address, dist::Distribution, obs::RVValue)::RVValue
    error("A guide program should not have observed values")
end

function sample(sampler::UniversalGuideSampler, addr::Address, dist::Distribution, obs::Nothing)::RVValue
    value = get!(sampler.X, addr, rand(dist)) # if there is a value in trace use it else sample
    sampler.W += logpdf(dist, value)
    return value
end

function param(sampler::UniversalGuideSampler, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())
    if !haskey(sampler.params_to_ix, addr)
        # add new parameter
        n = length(sampler.phi)
        ix = (n+1):(n+size)
        sampler.params_to_ix[addr] = ix
        sampler.constraints[addr] = constraint
        # all parameters are initialised to 0, vcat to preserve TrackedVector type
        if Tracker.istracked(sampler.phi)
            sampler.phi = vcat(sampler.phi, Tracker.param(zeros(size)))
        else
            sampler.phi = vcat(sampler.phi, zeros(eltype(sampler.phi), size))
        end
    end
    # inject parameter
    ix = sampler.params_to_ix[addr]
    if size == 1
        return constrain_param(constraint, sampler.phi[ix[1]])
    else
        return constrain_param(constraint, sampler.phi[ix])
    end
end

import TinyPPL.Distributions: VariationalDistribution

"""
Wraps `UniversalGuideSampler` and guide program.
Implements VariationalDistribution interface.
"""
struct UniversalGuide <: VariationalDistribution
    sampler::UniversalGuideSampler
    model::UniversalModel
    args::Tuple
    observations::Observations
end

# guide can be used for unconstrained and constrained logjoint
function make_guide(model::UniversalModel, args::Tuple)::UniversalGuide
    sampler = UniversalGuideSampler(Param2Ix(), zeros(0))
    return UniversalGuide(sampler, model, args, Observations())
end
export make_guide

import TinyPPL.Distributions: get_params
function get_params(q::UniversalGuide)::VariationalParameters
    return q.sampler.phi
end

import TinyPPL.Distributions: update_params
function update_params(guide::UniversalGuide, params::VariationalParameters)::VariationalDistribution
    # since UniversalGuideSampler is generic type, we freshly instantiate
    # q_ = update_params(q, no_grad(get_params(q))) before rand(q_) or rand(q)
    # can lead to descrepancies between the size of phi of q_ and q if params_to_ix are the same
    # -> copy params_to_ix
    new_sampler = UniversalGuideSampler(copy(guide.sampler.params_to_ix), params, copy(guide.sampler.constraints))
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


# mutable struct UniversalParameterTransformer <: UniversalSampler
#     phi::VariationalParameters
#     params_to_ix::Param2Ix
#     transformed_phi::VariationalParameters
#     function UniversalParameterTransformer(guide::UniversalGuide)
#         return new(guide.sampler.phi, guide.sampler.params_to_ix, similar(guide.sampler.phi))
#     end
# end

# const ParameterTransformer = Union{UniversalParameterTransformer, StaticParameterTransformer}

# function sample(sampler::ParameterTransformer, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
#     if !isnothing(obs)
#         return obs
#     end
#     return mean(dist)
# end

# function param(sampler::ParameterTransformer, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())
#     ix = sampler.params_to_ix[addr]
#     if size == 1
#         parameters = constrain_param(constraint, sampler.phi[ix[1]])
#     else
#         parameters = constrain_param(constraint, sampler.phi[ix])
#     end
#     sampler.transformed_phi[ix] .= parameters
#     return parameters
# end

"""
Extracts parameters of guide and transforms them to the specified *static* constraints.
UniversalParameterTransformer with dynamic constraints does not work,
since we do not know if all parameters are encountered in model execution.
"""
function get_constrained_parameters(guide::UniversalGuide)
    transformed_phi = similar(guide.sampler.phi)
    for (addr, ix) in guide.sampler.params_to_ix
        transformed_phi[ix] = constrain_param(guide.sampler.constraints[addr], guide.sampler.phi[ix])
    end
    return VIParameters(transformed_phi, guide.sampler.params_to_ix)
end

export get_constrained_parameters