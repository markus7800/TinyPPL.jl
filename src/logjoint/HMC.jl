import TinyPPL.Distributions: Normal
import ProgressLogging: @progress

# Several backends which compute gradient of logjoint.
import Tracker
function get_grad_U_tracker(logjoint::Function)
    function grad_U(X::Vector{Float64})
        X = Tracker.param(X)
        log_prob = logjoint(X)
        U = -log_prob
        Tracker.back!(U)
        return Tracker.grad(X)
    end
    return grad_U
end

import ForwardDiff
function get_grad_U_fwd_diff(logjoint::Function)
    function grad_U(X::Vector{Float64})
        grad = ForwardDiff.gradient(logjoint, X)
        return -grad # U = -logjoint
    end
    return grad_U
end

import ReverseDiff
function get_grad_U_rev_diff(logjoint::Function)
    function grad_U(X::Vector{Float64})
        grad = ReverseDiff.gradient(logjoint, X)
        return -grad # U = -logjoint
    end
    return grad_U
end

# The leapfrog integrated runs for L steps with stride eps.
# It has the property that leapfrog(*leapfrog(R, X, L, eps), L, eps) == (R, X)
function leapfrog(
        grad_U::Function,
        X::Vector{Float64}, P::Vector{Float64},
        L::Int, eps::Float64
    )

    P = P - eps/2 * grad_U(X)
    for _ in 1:(L-1)
        X = X + eps * P
        P = P - eps * grad_U(X)
    end
    X = X + eps * P
    P = P - eps/2 * grad_U(X)
    
    return X, -P
end

"""
Hamiltonian Monte Carlo with potential function -logjoint, number of variables `K`
trajectory length `L` and leapfrog step-size `eps`.
"""
function hmc_logjoint(logjoint::Function, K::Int, n_samples::Int, L::Int, eps::Float64;
    ad_backend::Symbol=:tracker, x_initial::Union{Nothing,Vector{Float64}}=nothing)
    if ad_backend == :tracker
        grad_U = get_grad_U_tracker(logjoint)
    elseif ad_backend == :forwarddiff
        grad_U = get_grad_U_fwd_diff(logjoint)
    elseif ad_backend == :reversediff
        grad_U = get_grad_U_rev_diff(logjoint)
    else
        error("Unkown ad backend $ad_backend.")
    end
    
    # initialise x0
    X_current = isnothing(x_initial) ? zeros(K) : x_initial
    log_prob_current = logjoint(X_current)
    U_current = -log_prob_current

    result = Array{Float64,2}(undef, K, n_samples)
    n_accepted = 0
    @progress for i in 1:n_samples
        P_current = rand(Normal(0., 1.),K)
        K_current = P_current'P_current / 2

        X_proposed, P_proposed = leapfrog(grad_U, X_current, P_current, L, eps)

        # Compute new kinetic and potential energy     
        K_proposed = P_proposed'P_proposed / 2
        log_prob_proposed = logjoint(X_proposed)
        U_proposed = -log_prob_proposed

        # With perfect precision the leapfrog integrator should preserve the energy and accept with probability 1.
        # But it is an approximation and we adjust with a metropolis hasting step
        if log(rand()) < (U_current - U_proposed + K_current - K_proposed) # -H(proposed) + H(current)
            U_current = U_proposed
            X_current = X_proposed
            n_accepted += 1
        end
        # Store regardless of acceptance
        result[:,i] = X_current
    end

    @info "HMC" n_accepted/n_samples

    return result
end

export hmc_logjoint