import Tracker
import Distributions
import LinearAlgebra
import PDMats
import Random

# for f in :[rand, randn, randexp].args
#     @eval Random.$f(rng::Random.AbstractRNG,::Type{Tracker.TrackedReal{T}}) where {T} = Tracker.param(Random.$f(rng,T))
# end

function advi_meanfield(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, K::Int)
    phi = zeros(2*K)

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        mu = phi_tracked[1:K]
        omega = phi_tracked[K+1:end]

        # estimate elbo
        elbo = 0.
        for _ in 1:L
            # reparametrisation trick
            eta = randn(K)
            zeta = @. exp(omega) * eta + mu
            elbo += logjoint(zeta)
        end
        elbo = elbo / L + sum(omega) # + K/2 * (log(2π) + 1)

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi_tracked)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    mu = phi[1:K]
    omega = phi[K+1:end]
    return mu, exp.(omega)
end

import LinearAlgebra: transpose, inv, LowerTriangular, Diagonal
function advi_fullrank(logjoint::Function, n_samples::Int, N::Int, learning_rate::Float64, K::Int)
    phi = vcat(zeros(K), reshape(LinearAlgebra.I(K),:))

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    mask = LinearAlgebra.LowerTriangular(trues(K,K))

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        mu = phi_tracked[1:K] # Tracked K-element Vector{Float64}
        # mu = convert(Vector{eltype(phi_tracked)}, phi_tracked[1:K]) # Vector{Tracker.TrackedReal{Float64}}
        A = reshape(phi_tracked[K+1:end],K,K) # Tracked K×K Matrix{Float64}
        L = A .* mask # Tracked K×K Matrix{Float64}
        # L = LinearAlgebra.LowerTriangular(convert(Matrix{eltype(A)}, A)) # Matrix{Tracker.TrackedReal{Float64}}

        # estimate elbo
        elbo = 0.
        for _ in 1:N
            # reparametrisation trick
            eta = randn(K)
            zeta = L*eta .+ mu
            elbo += logjoint(zeta)
        end
        elbo = elbo / N

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi_tracked)
        grad[K+1:end] += reshape(inv(Diagonal(Tracker.data(L))),:) # entropy

        # reset from gradient computation
        phi = Tracker.data(phi)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    mu = phi[1:K]
    L = reshape(phi[K+1:end],K,K)
    return mu, L
end


abstract type VariationalDistribution <: Distribution{Distributions.Multivariate, Distributions.Continuous} end

function update_params(q::VariationalDistribution, params::AbstractVector{<:Float64})::VariationalDistribution
    error("Not implemented.")
end
function nparams(q::VariationalDistribution)
    error("Not implemented.")
end
function init_params(q::VariationalDistribution)::AbstractVector{<:Float64}
    error("Not implemented.")
end
function rand_and_logpdf(q::VariationalDistribution)
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

function update_params(q::MeanFieldGaussian, params::AbstractVector{<:Float64})::VariationalDistribution
    K = length(q.mu)
    mu = params[1:K]
    omega = params[K+1:end]
    return MeanFieldGaussian(mu, exp.(omega))
end

function init_params(q::MeanFieldGaussian)::AbstractVector{<:Float64}
    return zeros(nparams(q))
end

function nparams(q::MeanFieldGaussian)
    return 2*length(q.mu)
end
function rand_and_logpdf(q::MeanFieldGaussian)
    K = length(q.mu)
    Z = randn(K)
    value = q.sigma .* Z .+ q.mu
    return value, -Z'Z/2 - K*log(sqrt(2π)) - log(prod(q.sigma))
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

function update_params(q::FullRankGaussian, params::AbstractVector{<:Float64})::VariationalDistribution
    K = length(q.base)
    mu = params[1:K]
    mu = convert(Vector{eltype(mu)}, mu)
    A = reshape(params[K+1:end], K, K)
    A = convert(Matrix{eltype(A)}, A) # Tracked K×K Matrix{Float64} -> K×K Matrix{Tracker.TrackedReal{Float64}}

    L = LinearAlgebra.LowerTriangular(A) # KxK LinearAlgebra.LowerTriangular{Tracker.TrackedReal{Float64}, Matrix{Tracker.TrackedReal{Float64}}}
    return FullRankGaussian(Distributions.MultivariateNormal(mu, PDMats.PDMat(LinearAlgebra.Cholesky(L))))
end

function nparams(q::FullRankGaussian)
    return sum(length, Distributions.params(q.base))
end

function init_params(q::FullRankGaussian)::AbstractVector{<:Float64}
    K = length(q.base)
    return vcat(zeros(K), reshape(LinearAlgebra.I(K),:))
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

function Distributions.entropy(q::FullRankGaussian)
    K = length(q.base)
    L = q.base.Σ.chol.L
    return K/2*(log(2π) + 1) + log(abs(prod(LinearAlgebra.diag(L))))
end

abstract type ELBOEstimator end
struct MonteCarloELBO <: ELBOEstimator end
function estimate_elbo(::MonteCarloELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    for _ in 1:L
        # implicit reparametrisation trick (if we get gradients)
        zeta, lpq = rand_and_logpdf(q)
        elbo += logjoint(zeta) - lpq
    end
    elbo = elbo / L
    return elbo
end

struct RelativeEntropyELBO <: ELBOEstimator end
function estimate_elbo(::RelativeEntropyELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    for _ in 1:L
        # implicit reparametrisation trick (if we get gradients)
        zeta, _ = rand_and_logpdf(q)
        elbo += logjoint(zeta)
    end
    elbo = elbo / L + Distributions.entropy(q)
    return elbo
end

function advi(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    phi = init_params(q)

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        q = update_params(q, phi_tracked)

        # estimate elbo
        elbo = estimate_elbo(estimator, logjoint, q, L)

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi_tracked)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    return update_params(q, phi)
end

export advi, advi_meanfield, advi_fullrank, MeanFieldGaussian, FullRankGaussian, MonteCarloELBO, RelativeEntropyELBO