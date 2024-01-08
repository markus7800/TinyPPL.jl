"""
Accumulates the log density log p(X,Y) for input trace `X`.
"""
mutable struct StaticLogJointSampler{T,V} <: StaticSampler
    W::T
    addresses_to_ix::Addr2Ix
    X::V
    # type-optimised constructors
    function StaticLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where V <: Vector{Float64}
        return new{Float64,V}(0., addresses_to_ix, X)
    end
    function StaticLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where V <: Tracker.TrackedVector{Float64, Vector{Float64}}
        return new{Tracker.TrackedReal{Float64},V}(0., addresses_to_ix, X)
    end
    function StaticLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where V <: Vector{Tracker.TrackedReal{Float64}}
        return new{Tracker.TrackedReal{Float64},V}(0., addresses_to_ix, X)
    end
    # fall back
    function StaticLogJointSampler(addresses_to_ix::Addr2Ix, X::AbstractStaticTrace)
        return new{Real,typeof(X)}(0., addresses_to_ix, X)
    end
end

function sample(sampler::StaticLogJointSampler, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    value = sampler.X[sampler.addresses_to_ix[addr]]
    sampler.W += logpdf(dist, value)
    return value
end

"""
Transforms a static model to a function which takes as input an static trace `X`
and returns the log density log p(X,Y).
Closes over args and observations Y.
"""
function make_logjoint(model::StaticModel, args::Tuple, observations::Observations)
    addresses_to_ix = get_address_to_ix(model, args, observations)
    function logjoint(X::AbstractStaticTrace)
        sampler = StaticLogJointSampler(addresses_to_ix, X)
        model(args, sampler, observations)
        return sampler.W
    end
    return logjoint, addresses_to_ix
end



import Tracker
"""
Accumulates the log density log p(T(X),Y) + log abs det ∇T(X) for input trace `X`.
The values of continuous distributions are assumed to be unconstrained and are mapped
to the support via a transformation T.
"""
mutable struct StaticUnconstrainedLogJointSampler{T,V} <: StaticSampler
    W::T
    addresses_to_ix::Addr2Ix
    X::V
    # type-optimised constructors
    function StaticUnconstrainedLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where V <: Vector{Float64}
        return new{Float64,V}(0., addresses_to_ix, X)
    end
    function StaticUnconstrainedLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where V <: Tracker.TrackedVector{Float64, Vector{Float64}}
        return new{Tracker.TrackedReal{Float64},V}(0., addresses_to_ix, X)
    end
    function StaticUnconstrainedLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where V <: Vector{Tracker.TrackedReal{Float64}}
        return new{Tracker.TrackedReal{Float64},V}(0., addresses_to_ix, X)
    end
    # fall back
    function StaticUnconstrainedLogJointSampler(addresses_to_ix::Addr2Ix, X::AbstractStaticTrace)
        return new{Real,typeof(X)}(0., addresses_to_ix, X)
    end
end
function sample(sampler::StaticUnconstrainedLogJointSampler, addr::Address, dist::Distributions.DiscreteDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    value = sampler.X[sampler.addresses_to_ix[addr]]
    sampler.W += logpdf(dist, value)
    return value
end
function sample(sampler::StaticUnconstrainedLogJointSampler, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    unconstrained_value = sampler.X[sampler.addresses_to_ix[addr]]
    transformed_dist = to_unconstrained(dist)
    sampler.W += logpdf(transformed_dist, unconstrained_value)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

"""
Transforms a universal model to a function which takes as input an universal trace `X`
and returns the log density log p(T(X),Y) + log abs det ∇T(X).
Closes over args and observations Y.
"""
function make_unconstrained_logjoint(model::StaticModel, args::Tuple, observations::Observations)
    addresses_to_ix = get_address_to_ix(model, args, observations)

    function logjoint(X::AbstractStaticTrace)
        sampler = StaticUnconstrainedLogJointSampler(addresses_to_ix, X)
        model(args, sampler, observations)
        return sampler.W
    end

    return logjoint, addresses_to_ix
end


import TinyPPL.Distributions: Transform, transform_to, to_unconstrained, support

mutable struct StaticConstraintTransformer{T} <: StaticSampler
    addresses_to_ix::Addr2Ix
    X::T
    Y::T # inplace transformation Y = X is allowed, as the value of X[i] is only read once.
    to::Symbol
    function StaticConstraintTransformer(addresses_to_ix::Addr2Ix, X::T, Y::T; to::Symbol) where T <: AbstractStaticTrace
        @assert to in (:constrained, :unconstrained)
        return new{T}(addresses_to_ix, X, Y, to)
    end
end 
function sample(sampler::StaticConstraintTransformer, addr::Address, dist::Distributions.DiscreteDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    i = sampler.addresses_to_ix[addr]
    value = sampler.X[i]
    sampler.Y[i] = value
    return value
end

function sample(sampler::StaticConstraintTransformer, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    transformed_dist = to_unconstrained(dist)
    i = sampler.addresses_to_ix[addr]
    if sampler.to == :unconstrained
        constrained_value = sampler.X[i]
        unconstrained_value = transformed_dist.T(constrained_value)
        sampler.Y[i] = unconstrained_value
    else # samper.to == :constrained
        unconstrained_value = sampler.X[i]
        constrained_value = transformed_dist.T_inv(unconstrained_value)
        sampler.Y[i] = constrained_value
    end
    return constrained_value
end

function transform_to_constrained!(X::AbstractStaticTrace, model::StaticModel, args::Tuple, constraints::Observations, addresses_to_ix::Addr2Ix)::Tuple{<:AbstractStaticTrace,Any}
    sampler = StaticConstraintTransformer(addresses_to_ix, X, X, to=:constrained)
    retval = model(args, sampler, constraints)
    return sampler.Y, retval
end

function transform_to_unconstrained!(X::AbstractStaticTrace, model::StaticModel, args::Tuple, constraints::Observations, addresses_to_ix::Addr2Ix)::Tuple{<:AbstractStaticTrace,Any}
    sampler = StaticConstraintTransformer(addresses_to_ix,  X, X, to=:unconstrained)
    retval = model(args, sampler, constraints)
    return sampler.Y, retval
end

function transform_to_constrained(X::AbstractStaticTrace, model::StaticModel, args::Tuple, constraints::Observations, addresses_to_ix::Addr2Ix)::Tuple{<:AbstractStaticTrace,Any}
    sampler = StaticConstraintTransformer(addresses_to_ix, X, similar(X), to=:constrained)
    retval = model(args, sampler, constraints)
    return sampler.Y, retval
end

function transform_to_unconstrained(X::AbstractStaticTrace, model::StaticModel, args::Tuple, constraints::Observations, addresses_to_ix::Addr2Ix)::Tuple{<:AbstractStaticTrace,Any}
    sampler = StaticConstraintTransformer(addresses_to_ix,  X, similar(X), to=:unconstrained)
    retval = model(args, sampler, constraints)
    return sampler.Y, retval
end

export transform_to_constrained, transform_to_unconstrained