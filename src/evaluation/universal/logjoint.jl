import Distributions

"""
Accumulates the log density log p(X,Y) for input trace `X`.
"""
mutable struct LogJointSampler <: UniversalSampler
    W::Real # log p(X,Y) tracked or untracked
    X::AbstractUniversalTrace
    function LogJointSampler(X::AbstractUniversalTrace)
        return new(0.,X)
    end
end
function sample(sampler::LogJointSampler, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    return value
end

"""
Transforms a universal model to a function which takes as input an universal trace `X`
and returns the log density log p(X,Y).
Closes over args and observations Y.
"""
function make_logjoint(model::UniversalModel, args::Tuple, observations::Observations)
    return function logjoint(X::AbstractUniversalTrace)
        sampler = LogJointSampler(X)
        model(args, sampler, observations)
        return sampler.W
    end
end

"""
Accumulates the log density log p(T(X),Y) + log abs det ∇T(X) for input trace `X`.
The values of continuous distributions are assumed to be unconstrained and are mapped
to the support via a transformation T.
"""
mutable struct UnconstrainedLogJointSampler <: UniversalSampler
    W::Real # log p(T(X),Y) + log abs det ∇T(X) tracked or untracked
    X::AbstractUniversalTrace
    function UnconstrainedLogJointSampler(X::AbstractUniversalTrace)
        return new(0.,X)
    end
end

function sample(sampler::UnconstrainedLogJointSampler, addr::Address, dist::Distributions.DiscreteDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    value = sampler.X[addr]
    sampler.W += logpdf(dist, value)
    return value
end

function sample(sampler::UnconstrainedLogJointSampler, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    # map unconstrained value to support
    unconstrained_value = sampler.X[addr]
    transformed_dist = to_unconstrained(dist) # T
    sampler.W += logpdf(transformed_dist, unconstrained_value) # log p(T^{-1}(X)) + log abs det ∇T^{-1}(X)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

"""
Transforms a universal model to a function which takes as input an universal trace `X`
and returns the log density log p(T(X),Y) + log abs det ∇T(X).
Closes over args and observations Y.
"""
function make_unconstrained_logjoint(model::UniversalModel, args::Tuple, observations::Dict)
    return function unconstrained_logjoint(X::AbstractUniversalTrace)
        sampler = UnconstrainedLogJointSampler(X)
        model(args, sampler, observations)
        return sampler.W
    end
end



"""
Transforms the values in trace `X` to either the original constraint model,
or to the unconstrained model.
We can only transform the values by execution the model instead of applying
the same transformations to all traces, since the values of `X` may affect the
transforms that are used.
E.g. a ~ Uniform(-1,1) b ~ Uniforma(-a, a)
transform for b depends on the value of a.

Is deterministic, does not use rand().

`X` can have more addresses than `Y` if a variational distribution samples too
much addresses, e.g. UniversalMeanField.
"""
mutable struct UniversalConstraintTransformer <: UniversalSampler
    X::AbstractUniversalTrace   # original trace
    Y::UniversalTrace   # transformed trace
    to::Symbol          # :constrained or :unconstrained
    function UniversalConstraintTransformer(X::AbstractUniversalTrace, to::Symbol)
        @assert to in (:constrained, :unconstrained)
        return new(X, UniversalTrace(), to)
    end
end 
function sample(sampler::UniversalConstraintTransformer, addr::Address, dist::Distributions.DiscreteDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    sampler.Y[addr] = sampler.X[addr]
    return sampler.Y[addr]
end
function sample(sampler::UniversalConstraintTransformer, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end
    transformed_dist = to_unconstrained(dist)
    if sampler.to == :unconstrained
        constrained_value = sampler.X[addr]
        unconstrained_value = transformed_dist.T(constrained_value)
        sampler.Y[addr] = unconstrained_value
    else # samper.to == :constrained
        unconstrained_value = sampler.X[addr]
        constrained_value = transformed_dist.T_inv(unconstrained_value)
        sampler.Y[addr] = constrained_value
    end
    return constrained_value
end

function transform_to_constrained(X::AbstractUniversalTrace, model::UniversalModel, args::Tuple, constraints::Observations)::Tuple{<:UniversalTrace,Any}
    sampler = UniversalConstraintTransformer(X, :constrained)
    retval = model(args, sampler, constraints)
    return sampler.Y, retval
end

function transform_to_unconstrained(X::AbstractUniversalTrace, model::UniversalModel, args::Tuple, constraints::Observations)::Tuple{<:UniversalTrace,Any}
    sampler = UniversalConstraintTransformer(X, :unconstrained)
    retval = model(args, sampler, constraints)
    return sampler.Y, retval
end

export transform_to_constrained, transform_to_unconstrained

# function transform_to_constrained(Xs::Vector{<:AbstractUniversalTrace}, model::UniversalModel, args::Tuple, constraints::Observations)::Tuple{Vector{UniversalTrace},Vector{Any}}
#     samples = Vector{UniversalTrace}(undef, length(Xs))
#     retvals = Vector{Any}(undef, length(Xs))
#     for i in eachindex(Xs)
#         @inbounds samples[i], retvals[i] = transform_to_constrained(Xs[i], model, args, constraints)
#     end
#     return samples, retvals
# end
