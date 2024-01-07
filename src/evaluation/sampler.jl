
const Address = Any
const RVValue = Real # currently only univariate distributions are supported
const Observations = Dict{Address, RVValue}
export Address, Observations

"""
Samplers facilitate a *contextualised execution* of the probabilistic program.
To accommodate this the samplers implement custom `sample` and/or `param` methods,
which control the behavior of sample and parameter statements in the program.
"""
abstract type Sampler end

function sample(sampler::Sampler, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    error("sample not implemented for $sampler.")
end

abstract type ParamConstraint end

"""
Internally parameters are always unconstrained values in (-∞, ∞).
For convenience, one may constrain the parameters to (0,∞) or (0,1) by passing a `constraint`.
This is often useful for distribution parameters.
You may instantiate also multiple parameters at once by passing a positive integer to the `size` argument.
"""
function param(sampler::Sampler, addr::Address; size::Int=1, constraint::ParamConstraint=Unconstrained())::AbstractArray{<:Real}
    error("param not implemented for $sampler.")
end

abstract type StaticSampler <: Sampler end
abstract type UniversalSampler <: Sampler end


function constrain_param(::ParamConstraint, x)
    error("Not implemented.")
end

# (-∞, ∞)
struct Unconstrained <: ParamConstraint end
constrain_param(::Unconstrained, x) = x

# (0, ∞)
struct Positive <: ParamConstraint end
constrain_param(::Positive, x) = exp.(x)

# (0, 1)
struct ZeroToOne <: ParamConstraint end
constrain_param(::ZeroToOne, x) = sigmoid.(x)
export Unconstrained, Positive, ZeroToOne

export sample, param