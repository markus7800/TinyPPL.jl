import Tracker
import PDMats
import Distributions
import LinearAlgebra
import Random
import TinyPPL.Distributions: ELBOEstimator, estimate_elbo, VariationalDistribution, update_params, get_params, rand_and_logpdf, logpdf, MeanFieldGaussian, FullRankGaussian
import TinyPPL: no_grad
# Fix merged to Tracker.jl
# for f in :[rand, randn, randexp].args
#     @eval Random.$f(rng::Random.AbstractRNG,::Type{Tracker.TrackedReal{T}}) where {T} = Tracker.param(Random.$f(rng,T))
# end

# TODO: backend for meanfield and fullrank

"""
ADVI with Gaussian meanfield approximation, fitted to `logjoint`.
Approximates ELBO gradient with closed form entropy.
Thus, is equivalent to advi_logjoint with MeanFieldGaussian and RelativeEntropyELBO
"""
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
    return MeanFieldGaussian(mu, omega, exp.(omega))
end

import LinearAlgebra: transpose, inv, LowerTriangular, Diagonal
"""
ADVI with Gaussian fullrank approximation, fitted to `logjoint`.
Approximates ELBO gradient with closed form entropy.
Thus, is equivalent to advi_logjoint with FullRankGaussian and RelativeEntropyELBO
"""
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
        A = reshape(phi_tracked[K+1:end],K,K) # Tracked K×K Matrix{Float64}
        L = A .* mask # Tracked K×K Matrix{Float64}

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
        grad[K+1:end] += reshape(inv(Diagonal(no_grad(L))),:) # entropy

        # reset from gradient computation
        phi = no_grad(phi)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    mu = phi[1:K]
    # L = reshape(phi[K+1:end],K,K)
    L = phi[K+1:end]
    return FullRankGaussian(mu, L)
end

function estimate_elbo_grad_tracker(logjoint::Function, phi::Vector{Float64}, q::VariationalDistribution, L::Int, estimator::ELBOEstimator)
    phi_tracked = Tracker.param(phi)
    q = update_params(q, phi_tracked)
    elbo = estimate_elbo(estimator, logjoint, q, L)
    Tracker.back!(elbo)
    grad = Tracker.grad(phi_tracked)
    return grad
end

# The ELBO estimate is stochastic, so the gradient has to be computed from a single evaluation
# (with a single set of random draws).
# ForwardDiff evaluates the function once per chunk of partials; by default the chunk size is at most 12,
# so for more parameters each chunk would see different random draws
# (and the old implementation, which only did the first chunk, returned a truncated gradient).
# We therefore use a single chunk containing all parameters.
function estimate_elbo_grad_forwarddiff(logjoint::Function, phi::Vector{Float64}, q::VariationalDistribution, L::Int, estimator::ELBOEstimator)
    f = phi -> estimate_elbo(estimator, logjoint, update_params(q, phi), L)
    # Chunk{N}() instead of Chunk(N), which would again cap the chunk size at 12
    cfg = ForwardDiff.GradientConfig(f, phi, ForwardDiff.Chunk{length(phi)}())
    return ForwardDiff.gradient(f, phi, cfg)

    # Worked out, ForwardDiff.gradient(f, phi, cfg) does the following (vector mode, single chunk):
    # phi_dual = cfg.duals                              # Vector of Dual numbers with length(phi) partials
    # ForwardDiff.seed!(phi_dual, phi, cfg.seeds)       # phi_dual[i] = Dual(phi[i], e_i)
    # q = update_params(q, phi_dual)
    # elbo = estimate_elbo(estimator, logjoint, q, L)   # Dual(elbo, ∇elbo)
    # grad = ForwardDiff.partials(elbo)                 # ∇elbo
    # (the previous implementation used the default GradientConfig(estimate_elbo, phi),
    #  whose chunk size is min(length(phi), 12), so the partials only covered the first chunk)
end

# ReverseDiff records the tape with one forward pass (and thus one set of random draws) and then runs the reverse pass.
# (The tape is not compiled / reused, since the random draws would be recorded as constants.)
function estimate_elbo_grad_reversediff(logjoint::Function, phi::Vector{Float64}, q::VariationalDistribution, L::Int, estimator::ELBOEstimator)
    f = phi -> estimate_elbo(estimator, logjoint, update_params(q, phi), L)
    return ReverseDiff.gradient(f, phi)

    # Worked out, ReverseDiff.gradient(f, phi) does the following (uses ReverseDiff internals):
    # cfg = ReverseDiff.GradientConfig(phi)
    # ReverseDiff.track!(cfg.input, phi)                # cfg.input is a TrackedArray recording onto cfg.tape
    # q = update_params(q, cfg.input)
    # elbo = estimate_elbo(estimator, logjoint, q, L)   # forward pass, records the tape
    # tape = ReverseDiff._GradientTape(f, cfg.input, elbo, cfg.tape)
    # grad = ReverseDiff.construct_result(ReverseDiff.input_hook(tape))
    # ReverseDiff.seeded_reverse_pass!(grad, tape)      # seeds ∂elbo/∂elbo = 1 and backpropagates into grad
end

"""
ADVI with arbitary VariationalDistribution approximation, fitted to `logjoint`.
Approximates ELBO gradient with ELBOEstimator.
"""
function advi_logjoint(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator; ad_backend::Symbol=:tracker)
    phi = no_grad(get_params(q))

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    # TODO: do ad_backend::Val{:tracker} instead?
    if ad_backend == :tracker
        estimate_elbo_grad = estimate_elbo_grad_tracker
    elseif ad_backend == :forwarddiff
        estimate_elbo_grad = estimate_elbo_grad_forwarddiff
    elseif ad_backend == :reversediff
        estimate_elbo_grad = estimate_elbo_grad_reversediff
    else
        error("Unkown ad backend $ad_backend.")
    end

    @progress for i in 1:n_samples
        grad = estimate_elbo_grad(logjoint, phi, q, L, estimator)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    return update_params(q, phi)
end

export advi_logjoint, advi_meanfield_logjoint, advi_fullrank_logjoint