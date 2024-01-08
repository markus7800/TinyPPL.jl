"""
This file includes various ELBO estimators.
"""


abstract type ELBOEstimator end
struct MonteCarloELBO <: ELBOEstimator end # requires differentiable model
function estimate_elbo(::MonteCarloELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    for _ in 1:L
        # implicit reparametrisation trick (if we get gradients)
        zeta, lpq = rand_and_logpdf(q)
        lpp = logjoint(zeta)
        elbo +=  lpp - lpq
    end
    elbo = elbo / L
    return elbo
end

struct RelativeEntropyELBO <: ELBOEstimator end # requires differentiable model and closed form entropy
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

struct ReinforceELBO <: ELBOEstimator end # does not require differentiable model
# this is a special case of the Reinforce gradient approximation where E_q(z|ϕ)[∇f_ϕ(z)] = 0
function estimate_elbo(::ReinforceELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    q_ = update_params(q, no_grad(get_params(q)))
    for _ in 1:L
        zeta = rand(q_)
        lpq = logpdf(q, zeta) # tracked q, untracked zeta
        no_grad_elbo = logjoint(zeta) - no_grad(lpq)
        @assert !Tracker.istracked(no_grad_elbo) zeta
        @assert Tracker.istracked(lpq)
        # inject log Q gradient
        elbo += no_grad_elbo * lpq + no_grad_elbo * (1 - no_grad(lpq))
    end
    elbo = elbo / L
    return elbo
end

# struct ReinforceSurrogateELBO <: ELBOEstimator end
# function estimate_elbo(::ReinforceSurrogateELBO, logjoint::Function, q::VariationalDistribution, L::Int)
#     surrogate_elbo = 0.
#     for _ in 1:L
#         zeta = no_grad(rand(q))
#         lpq = logpdf(q, zeta)
#         elbo = logjoint(zeta) - lpq
#         surrogate_elbo += no_grad(elbo) * lpq + elbo
#     end
#     surrogate_elbo = surrogate_elbo / L
#     return surrogate_elbo
# end

struct PathDerivativeELBO <: ELBOEstimator end # requires differentiable model
function estimate_elbo(::PathDerivativeELBO, logjoint::Function, q::VariationalDistribution, L::Int)
    elbo = 0.
    q_ = update_params(q, no_grad(get_params(q)))
    # println(get_params(q))
    for _ in 1:L
        zeta = rand(q)
        # println("zeta: ", zeta)
        elbo += logjoint(zeta) - logpdf(q_, zeta) # tracked zeta, untracked q
    end
    elbo = elbo / L
    return elbo
end

export MonteCarloELBO, RelativeEntropyELBO, ReinforceELBO, PathDerivativeELBO