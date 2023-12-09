import Distributions
import PDMats
import LinearAlgebra

abstract type VariationalDistribution end # <: Distribution{Distributions.Multivariate, Distributions.Continuous} end
export VariationalDistribution

function initial_params(q::VariationalDistribution)::AbstractVector{<:Float64}
    error("Not implemented.")
end
function get_params(q::VariationalDistribution)::AbstractVector{<:Real}
    error("Not implemented.")
end
function update_params(q::VariationalDistribution, params::AbstractVector{<:Real})::VariationalDistribution
    error("Not implemented.")
end
function rand_and_logpdf(q::VariationalDistribution)
    error("Not implemented.")
end
function Distributions.rand(q::VariationalDistribution)
    error("Not implemented.")
end
function Distributions.rand(q::VariationalDistribution, n::Int)
    error("Not implemented.")
end
function Distributions.entropy(q::VariationalDistribution)
    error("Not implemented.")
end

struct MeanFieldGaussian <: VariationalDistribution
    mu::AbstractVector{<:Real}
    sigma::AbstractVector{<:Real}
end

function MeanFieldGaussian(K::Int)
    return MeanFieldGaussian(zeros(K), ones(K))
end

function initial_params(q::MeanFieldGaussian)::AbstractVector{<:Float64}
    return zeros(2*length(q.mu))
end

function update_params(q::MeanFieldGaussian, params::AbstractVector{<:Real})::MeanFieldGaussian
    K = length(q.mu)
    mu = params[1:K]
    omega = params[K+1:end]
    return MeanFieldGaussian(mu, exp.(omega))
end

function rand_and_logpdf(q::MeanFieldGaussian)
    K = length(q.mu)
    Z = randn(K)
    value = q.sigma .* Z .+ q.mu
    return value, -Z'Z/2 - K*log(sqrt(2π)) - log(prod(q.sigma))
end
function Distributions.rand(q::MeanFieldGaussian)
    K = length(q.mu)
    Z = randn(K)
    return q.sigma * Z + q.mu
end
function Distributions.rand(q::MeanFieldGaussian, n::Int)
    K = length(q.mu)
    Z = randn(K, n)
    return q.sigma .* Z .+ q.mu
end

function Distributions.entropy(q::MeanFieldGaussian)
    return sum(log, q.sigma) + length(q.mu)/2 * (log(2π) + 1)
end

struct FullRankGaussian <: VariationalDistribution
    base::Distributions.MultivariateNormal
end

function FullRankGaussian(K::Int)
    return FullRankGaussian(Distributions.MultivariateNormal(zeros(K), LinearAlgebra.diagm(ones(K))))
end

function initial_params(q::FullRankGaussian)::AbstractVector{<:Float64}
    K = length(q.base)
    return vcat(zeros(K), reshape(LinearAlgebra.I(K),:))
end

function update_params(q::FullRankGaussian, params::AbstractVector{<:Real})::FullRankGaussian
    K = length(q.base)
    mu = params[1:K]
    mu = convert(Vector{eltype(mu)}, mu)
    A = reshape(params[K+1:end], K, K)
    A = convert(Matrix{eltype(A)}, A) # Tracked K×K Matrix{Float64} -> K×K Matrix{Tracker.TrackedReal{Float64}}

    L = LinearAlgebra.LowerTriangular(A) # KxK LinearAlgebra.LowerTriangular{Tracker.TrackedReal{Float64}, Matrix{Tracker.TrackedReal{Float64}}}
    return FullRankGaussian(Distributions.MultivariateNormal(mu, PDMats.PDMat(LinearAlgebra.Cholesky(L))))
end

function rand_and_logpdf(q::FullRankGaussian)
    # K = length(q.base)
    # L = q.base.Σ.chol.L
    # eta = randn(K)
    # zeta = L*eta .+ q.base.μ
    # value = zeta
    # now works with fixed Tracker randn
    value = rand(q.base)
    return value, Distributions.logpdf(q.base, value)
end
function Distributions.rand(q::FullRankGaussian)
    return rand(q.base)
end
function Distributions.rand(q::FullRankGaussian, n::Int)
    return rand(q.base, n)
end

function Distributions.entropy(q::FullRankGaussian)
    K = length(q.base)
    L = q.base.Σ.chol.L
    return K/2*(log(2π) + 1) + log(abs(prod(LinearAlgebra.diag(L))))
end

abstract type VariationalWrappedDistribution <: VariationalDistribution end

# function rand_and_logpdf(q::VariationalWrappedDistribution)
#     value = rand(q.base)
#     return value, Distributions.logpdf(q.base, value)
# end
function Distributions.rand(q::VariationalWrappedDistribution)
    return Distributions.rand(q.base)
end
function Distributions.rand(q::VariationalWrappedDistribution, n::Int)
    return Distributions.rand(q.base, n)
end
function Distributions.logpdf(q::VariationalWrappedDistribution, x::Real)
    return Distributions.logpdf(q.base, x)
end
struct VariationalNormal{T} <: VariationalWrappedDistribution where T <: Real
    base::Distributions.Normal{T}
    log_σ::T
end
function VariationalNormal()
    return VariationalNormal{Float64}(Distributions.Normal(), 0.)
end
function initial_params(q::VariationalNormal)::AbstractVector{<:Float64}
    return [0., 0.]
end
function get_params(q::VariationalNormal)::AbstractVector{<:Real}
    return [q.base.μ, q.log_σ]
end
function update_params(q::VariationalNormal, params::AbstractVector{<:Real})::VariationalNormal
    return VariationalNormal(Distributions.Normal(params[1], exp(params[2])), params[2])
end
function logpdf_param_grads(q::VariationalNormal, x::Real)
    z = (x - q.base.μ) / q.base.σ
    ∇μ = z / q.base.σ
    ∇σ = -1. / q.base.σ + abs2(z) / q.base.σ
    return [∇μ, ∇σ * exp(q.log_σ)]
end

struct VariationalGeometric{T} <: VariationalWrappedDistribution where T <: Real
    base::Distributions.Geometric{T}
    inv_sigmoid_p::T
end
function VariationalGeometric()
    return VariationalGeometric{Float64}(Distributions.Geometric(0.5), 0.)
end
function initial_params(q::VariationalGeometric)::AbstractVector{<:Float64}
    return [0.]
end
function get_params(q::VariationalGeometric)::AbstractVector{<:Real}
    return [q.inv_sigmoid_p]
end
function update_params(q::VariationalGeometric, params::AbstractVector{<:Real})::VariationalGeometric
    return VariationalGeometric(Distributions.Geometric(sigmoid(params[1])), params[1])
end
function logpdf_param_grads(q::VariationalGeometric, x::Real)
    p = q.base.p
    ∇p = x >= 0 ? 1/p - (1/(1-p) * x) : 0.
    return [∇p * ∇sigmoid(q.inv_sigmoid_p)]
end

# import ..Distributions: Transform, in_domain, log_abs_det_jacobian

# struct TransformedVariationalWrappedDistribution <: VariationalDistribution
#     base::VariationalWrappedDistribution
#     T::Transform
#     T_inv::Transform
# end
# function TransformedVariationalWrappedDistribution(base::VariationalWrappedDistribution, T::Transform)
#     return TransformedVariationalWrappedDistribution(base, T, inv(T))
# end
# function Distributions.rand(q::TransformedVariationalWrappedDistribution)
#     return q.T(Distributions.rand(q.base))
# end
# function Distributions.rand(q::TransformedVariationalWrappedDistribution, n::Int)
#     return q.T.(Distributions.rand(q.base, n))
# end
# function Distributions.logpdf(q::TransformedVariationalWrappedDistribution, y::Real)
#     if !in_domain(q.T_inv, y) # !(y in support(t))
#         return -Inf
#     end
#     x = q.T_inv(y)
#     return Distributions.logpdf(q.base, x) + log_abs_det_jacobian(q.T_inv, y)
# end
# function initial_params(q::TransformedVariationalWrappedDistribution)::AbstractVector{<:Float64}
#     return initial_params(q.base)
# end
# function get_params(q::TransformedVariationalWrappedDistribution)::AbstractVector{<:Real}
#     return get_params(q.base)
# end
# function update_params(q::TransformedVariationalWrappedDistribution, params::AbstractVector{<:Real})::TransformedVariationalWrappedDistribution
#     return TransformedVariationalWrappedDistribution(update_params(q.base, params), q.T, q.T_inv)
# end
# function logpdf_param_grads(q::TransformedVariationalWrappedDistribution, y::Real)
#     x = q.T_inv(y)
#     return logpdf_param_grads(q.base, x)
# end

export MeanFieldGaussian, FullRankGaussian
export VariationalNormal, VariationalGeometric