import Tracker
import ..Distributions: Normal
import ProgressLogging: @progress

function get_grad_U(logjoint::Function)
    function grad_U(X::AbstractVector{<:Real})
        X = Tracker.param(Tracker.data(X))
        log_prob = logjoint(X)
        U = -log_prob
        Tracker.back!(U)
        return Tracker.grad(X)
    end
    return grad_U
end

# The leapfrog integrated runs for L steps with stride eps.
# It has the property that leapfrog(*leapfrog(R, X, L, eps), L, eps) == (R, X)
function leapfrog(
        grad_U::Function,
        X::AbstractVector{<:Real}, P::AbstractVector{<:Real},
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

function hmc_logjoint(logjoint::Function, K::Int, n_samples::Int, L::Int, eps::Float64)
    grad_U = get_grad_U(logjoint)
    
    # initialise x0
    X_current = zeros(K)
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
        if log(rand()) < (U_current - U_proposed + K_current - K_proposed)
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