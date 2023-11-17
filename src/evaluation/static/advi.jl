import Tracker
import Distributions
import LinearAlgebra
import PDMats

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

        # reparametrisation trick
        eta = randn(K)
        zeta = @. exp(omega) * eta + mu

        # estimate elbo
        elbo = 0.
        for _ in 1:L
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

    #mask = LinearAlgebra.I(K)
    mask = LinearAlgebra.LowerTriangular(trues(K,K))

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        #mu = convert(Vector{eltype(phi_tracked)}, phi_tracked[1:K])
        mu = phi_tracked[1:K]
        A = reshape(phi_tracked[K+1:end],K,K)
        L = A .* mask
        # L = LinearAlgebra.LowerTriangular(convert(Matrix{eltype(A)}, A))
        # println(mu, ", ", L)

        # reparametrisation trick
        eta = randn(K)
        zeta = L*eta .+ mu

        # estimate elbo
        elbo = 0.
        for _ in 1:N
            elbo += logjoint(zeta)
        end
        elbo = elbo / N

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi_tracked)
        # println(Tracker.data(L))
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

function update_q(q::VariationalDistribution, phi::AbstractVector{<:Float64})::VariationalDistribution
    error("Not implemented.")
end
function nparams(q::VariationalDistribution)
    return sum(length, Distributions.params(q.base))
end
function init_phi(q::VariationalDistribution)::AbstractVector{<:Float64}
    return zeros(nparams(q))
end
function Base.rand(q::VariationalDistribution)
    return rand(q.base)
end
function Base.rand(q::VariationalDistribution, n::Int)
    return rand(q.base, n)
end
function Distributions.entropy(q::VariationalDistribution)
    return Distributions.entropy(q.base)
end   

struct MeanFieldGaussian <: VariationalDistribution
    mu::AbstractVector{<:Real}
    sigma::AbstractVector{<:Real}
end

function MeanFieldGaussian(K::Int)
    return MeanFieldGaussian(zeros(K), ones(K))
end

function update_q(q::MeanFieldGaussian, phi::AbstractVector{<:Float64})::VariationalDistribution
    K = length(q.mu)
    mu = phi[1:K]
    omega = phi[K+1:end]
    return MeanFieldGaussian(mu, exp.(omega))
end
function nparams(q::MeanFieldGaussian)
    return 2*length(q.mu)
end
function Base.rand(q::MeanFieldGaussian)
    return q.sigma .* randn(length(q.mu)) .+ q.mu
end
function Base.rand(q::MeanFieldGaussian, n::Int)
    return q.sigma .* randn(length(q.mu), n) .+ q.mu
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

function update_q(q::FullRankGaussian, phi::AbstractVector{<:Float64})::VariationalDistribution
    K = length(q.base)
    mu = phi[1:K]
    A = reshape(phi[K+1:end], K, K)
    A = convert(Matrix{eltype(A)}, A) # Tracked K×K Matrix{Float64} -> K×K Matrix{Tracker.TrackedReal{Float64}}

    L = LinearAlgebra.LowerTriangular(A)
    return FullRankGaussian(Distributions.MultivariateNormal(mu, PDMats.PDMat(LinearAlgebra.Cholesky(L))))
end

function init_phi(q::FullRankGaussian)::AbstractVector{<:Float64}
    K = length(q.base)
    return vcat(zeros(K), reshape(LinearAlgebra.I(K),:))
end

function Distributions.entropy(q::FullRankGaussian)
    return sum(log, q.sigma) + length(q.mu)/2 * (log(2π) + 1)
end


function advi(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution)
    phi = init_phi(q)

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        q = update_q(q, phi_tracked)

        # implicit reparametrisation trick (if we get gradients)
        zeta = rand(q)

        # estimate elbo
        elbo = 0.
        for _ in 1:L
            elbo += logjoint(zeta)
        end
        elbo = elbo / L + Distributions.entropy(q)

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi_tracked)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    return update_q(q, phi)
end

export advi, advi_meanfield, advi_fullrank, MeanFieldGaussian, FullRankGaussian