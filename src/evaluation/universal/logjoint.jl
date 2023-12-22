import Distributions

mutable struct LogJointSampler <: UniversalSampler
    W::Real # tracked or untracked
    X::Dict{Any,Real}
    function LogJointSampler(X::Dict{Any,Real})
        return new(0.,X)
    end
end
function sample(sampler::LogJointSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    return value
end

function make_logjoint(model::UniversalModel, args::Tuple, observations::Dict)
    return function logjoint(X::Dict{Any,Real})
        sampler = LogJointSampler(X)
        model(args, sampler, observations)
        return sampler.W
    end
end

mutable struct UnconstrainedLogJointSampler <: UniversalSampler
    W::Real # tracked or untracked
    X::Dict{Any,Real}
    function UnconstrainedLogJointSampler(X::Dict{Any,Real})
        return new(0.,X)
    end
end

function sample(sampler::UnconstrainedLogJointSampler, addr::Any, dist::Distributions.DiscreteDistribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    return value
end

function sample(sampler::UnconstrainedLogJointSampler, addr::Any, dist::Distributions.ContinuousDistribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    unconstrained_value = sampler.X[addr]
    transformed_dist = to_unconstrained(dist)
    sampler.W += logpdf(transformed_dist, unconstrained_value)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

function make_unconstrained_logjoint(model::UniversalModel, args::Tuple, observations::Dict)
    return function unconstrained_logjoint(X::Dict{Any,Real})
        sampler = UnconstrainedLogJointSampler(X)
        model(args, sampler, observations)
        return sampler.W
    end
end
