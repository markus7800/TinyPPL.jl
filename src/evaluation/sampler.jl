
abstract type Sampler end

function sample(sampler::Sampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    error("sample not implemented for $sampler.")
end

abstract type ParamConstraint end
function transform(::ParamConstraint, x)
    error("Not implemented.")
end
struct Unconstrained <: ParamConstraint end
transform(::Unconstrained, x) = x

struct Positive <: ParamConstraint end
transform(::Positive, x) = exp.(x)

struct ZeroToOne <: ParamConstraint end
transform(::ZeroToOne, x) = sigmoid.(x)
export Unconstrained, Positive, ZeroToOne


function param(sampler::Sampler, addr::Any, size::Int=1, constraint::ParamConstraint=Unconstrained())::AbstractArray{<:Real}
    error("param not implemented for $sampler.")
end

abstract type StaticSampler <: Sampler end
abstract type UniversalSampler <: Sampler end

export sample, param