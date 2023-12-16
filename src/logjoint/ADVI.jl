import Tracker
import Distributions
import LinearAlgebra
import Random
import ..Distributions: VariationalDistribution, initial_params, update_params, get_params, rand_and_logpdf, logpdf

# Fix merged to Tracker.jl
# for f in :[rand, randn, randexp].args
#     @eval Random.$f(rng::Random.AbstractRNG,::Type{Tracker.TrackedReal{T}}) where {T} = Tracker.param(Random.$f(rng,T))
# end

function advi_meanfield_logjoint(logjoint::Function, K::Int, n_samples::Int, L::Int, learning_rate::Float64)
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
function advi_fullrank_logjoint(logjoint::Function, K::Int, n_samples::Int, N::Int, learning_rate::Float64)
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

struct ReinforceELBO <: ELBOEstimator end
function estimate_elbo(::ReinforceELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    for _ in 1:L
        zeta = Tracker.data.(Tracker.data(rand(q))) # TODO
        lpq = logpdf(q, zeta)
        no_grad_elbo = logjoint(zeta) - Tracker.data(lpq)
        @assert !Tracker.istracked(no_grad_elbo) zeta
        @assert Tracker.istracked(lpq)
        # inject log Q gradient
        elbo += no_grad_elbo * lpq + no_grad_elbo * (1 - Tracker.data(lpq))
    end
    elbo = elbo / L
    return elbo
end

struct PathDerivativeELBO <: ELBOEstimator end
function estimate_elbo(::PathDerivativeELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    for _ in 1:L
        zeta = rand(q)
        q_ = update_params(q, Tracker.data.(Tracker.data(get_params(q)))) # TODO
        elbo += logjoint(zeta) - logpdf(q_, zeta)
    end
    elbo = elbo / L
    return elbo
end

function advi_logjoint(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    phi = initial_params(q)

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

export advi_logjoint, advi_meanfield_logjoint, advi_fullrank_logjoint, MonteCarloELBO, RelativeEntropyELBO, ReinforceELBO, PathDerivativeELBO