
mutable struct StaticLogJointSampler{T} <: StaticSampler
    W::T
    addresses_to_ix::Addr2Ix
    X::AbstractVector{T}
    function StaticLogJointSampler(addresses_to_ix::Addr2Ix, X::AbstractVector{T}) where T <: Real
        return new{T}(0., addresses_to_ix, X)
    end
end

function sample(sampler::StaticLogJointSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
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
        sampler = StaticLogJointSampler(addresses_to_ix, X)
        model(args, sampler, observations)
        return sampler.W
    end
    return logjoint, addresses_to_ix # TODO: type?
end


import ..Distributions: Transform, transform_to, to_unconstrained, support

mutable struct ConstraintTransformer{T} <: StaticSampler
    addresses_to_ix::Addr2Ix
    X::T
    Y::T
    to::Symbol
    function ConstraintTransformer(addresses_to_ix::Addr2Ix, X::T, Y::T; to::Symbol) where T <: AbstractVector{Float64}
        @assert to in (:constrained, :unconstrained)
        return new{T}(addresses_to_ix, X, Y, to)
    end
end 

function sample(sampler::ConstraintTransformer, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
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

mutable struct StaticUnconstrainedLogJointSampler{T,V} <: StaticSampler
    W::T
    addresses_to_ix::Addr2Ix
    X::V
    # TODO: handle T = Int
    function StaticUnconstrainedLogJointSampler(addresses_to_ix::Addr2Ix, X::V) where {T <: Real, V <: AbstractVector{T}}
        return new{eltype(V),V}(0., addresses_to_ix, X)
    end
end
function sample(sampler::StaticUnconstrainedLogJointSampler, addr::Any, dist::Distributions.DiscreteDistribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    value = sampler.X[sampler.addresses_to_ix[addr]]
    sampler.W += logpdf(dist, value)
    return value
end
function sample(sampler::StaticUnconstrainedLogJointSampler, addr::Any, dist::Distributions.ContinuousDistribution, obs::Union{Nothing, Real})::Real
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

struct UnconstrainedLogJoint
    addresses_to_ix::Addr2Ix
    logjoint::Function
    transform_to_constrained!::Function
    transform_to_unconstrained!::Function
end

function make_unconstrained_logjoint(model::StaticModel, args::Tuple, observations::Dict)
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(addresses)

    # TODO: only input matrix, if distributions (e.g. support) is also static
    function transform_to_constrained!(X::AbstractArray{Float64})
        if ndims(X) == 2
            for i in axes(X,2)
                X_i = view(X, :, i)
                sampler = ConstraintTransformer(addresses_to_ix, X_i, X_i, to=:constrained)
                model(args, sampler, observations)
            end
        else
            sampler = ConstraintTransformer(addresses_to_ix, X, X, to=:constrained)
            model(args, sampler, observations)
        end
        return X
    end

    function transform_to_unconstrained!(X::AbstractArray{Float64})
        if ndims(X) == 2
            for i in axes(X,2)
                X_i = view(X, :, i)
                sampler = ConstraintTransformer(addresses_to_ix,  X_i, X_i, to=:unconstrained)
                model(args, sampler, observations)
            end
        else
            sampler = ConstraintTransformer(addresses_to_ix, X, X, to=:unconstrained)
            model(args, sampler, observations)
        end
        return X
    end

    function logjoint(X::AbstractVector{<:Real})
        sampler = StaticUnconstrainedLogJointSampler(addresses_to_ix, X)
        model(args, sampler, observations)
        return sampler.W
    end

    return UnconstrainedLogJoint(addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained!)
end