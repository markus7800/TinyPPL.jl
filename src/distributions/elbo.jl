
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
        zeta = rand(q)
        elbo += logjoint(zeta)
    end
    elbo = elbo / L + Distributions.entropy(q)
    return elbo
end

struct ReinforceELBO <: ELBOEstimator end
function estimate_elbo(::ReinforceELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    q_ = update_params(q, no_grad(get_params(q)))
    for _ in 1:L
        zeta = rand(q_)
        lpq = logpdf(q, zeta)
        no_grad_elbo = logjoint(zeta) - no_grad(lpq)
        @assert !Tracker.istracked(no_grad_elbo) zeta
        @assert Tracker.istracked(lpq)
        # inject log Q gradient
        elbo += no_grad_elbo * lpq + no_grad_elbo * (1 - no_grad(lpq))
    end
    elbo = elbo / L
    return elbo
end

struct PathDerivativeELBO <: ELBOEstimator end
function estimate_elbo(::PathDerivativeELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    q_ = update_params(q, no_grad(get_params(q)))
    for _ in 1:L
        zeta = rand(q)
        elbo += logjoint(zeta) - logpdf(q_, zeta)
    end
    elbo = elbo / L
    return elbo
end

export MonteCarloELBO, RelativeEntropyELBO, ReinforceELBO, PathDerivativeELBO