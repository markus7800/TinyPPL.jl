# Linear Gaussian state space model (static/lgss.jl, graph/lgss.jl):
#
#   x_0 = 0, x_t ~ Normal(a x_{t-1}, σ_v), y_t ~ Normal(x_t, σ_e) for t = 1, ..., T.
#
# The data y (and the ground truth x_gt) is simulated from the model for 400 steps, scripts may use a prefix.
# Everything is jointly Gaussian, so the posterior p(x_1:T | y_1:T) and the evidence p(y_1:T) are known in closed form.

import Distributions: MvNormal, Normal, logpdf, mean, std
import LinearAlgebra: I, diag
import Random

const a = 0.9
const σ_v = 0.32
const σ_e = 1.
function simulate_lgss(T; seed=0)
    rng = Random.Xoshiro(seed)
    x_gt = zeros(T)
    y = zeros(T)
    x = 0.
    for t in 1:T
        x = rand(rng, Normal(a * x, σ_v))
        x_gt[t] = x
        y[t] = rand(rng, Normal(x, σ_e))
    end
    return x_gt, y
end
const x_gt, y = simulate_lgss(400)

# x_t = v_t + a v_{t-1} + ... + a^{t-1} v_1 with independent v_i ~ Normal(0, σ_v), so
# Var(x_t) = σ_v^2 (1 - a^{2t}) / (1 - a^2) and Cov(x_s, x_t) = a^{|t-s|} Var(x_min(s,t)).
# y = x + noise, so Cov(x, y) = Cov(x, x) and Cov(y, y) = Cov(x, x) + σ_e^2 I.
# Returns the posterior mean and standard deviation of x_1:T and the log evidence log p(y_1:T).
function lgss_exact(y)
    T = length(y)
    var_x(t) = σ_v^2 * (1 - a^(2t)) / (1 - a^2)
    Σ_x = [a^abs(t - s) * var_x(min(s, t)) for s in 1:T, t in 1:T]
    Σ_y = Σ_x + σ_e^2 * I
    μ_post = Σ_x * (Σ_y \ y)
    Σ_post = Σ_x - Σ_x * (Σ_y \ Σ_x)
    log_evidence = logpdf(MvNormal(zeros(T), Σ_y), y)
    return μ_post, sqrt.(diag(Σ_post)), log_evidence
end

# The locally optimal proposal for SMC: p(x_t | x_{t-1}, y_t) ∝ Normal(x_t; a x_{t-1}, σ_v) Normal(y_t; x_t, σ_e)
# is again Gaussian (conjugacy), with
#   mean = a x_{t-1} + σ_v^2 / (σ_v^2 + σ_e^2) (y_t - a x_{t-1}) and variance = σ_v^2 σ_e^2 / (σ_v^2 + σ_e^2).
# Since x_t only depends on x_{t-1} and y_t only on x_t, this equals p(x_t | x_1:t-1, y_1:t).
optimal_proposal(x_last, y_t) = Normal(a * x_last + σ_v^2 / (σ_v^2 + σ_e^2) * (y_t - a * x_last), sqrt(σ_v^2 * σ_e^2 / (σ_v^2 + σ_e^2)))

# Fraction of particle Gibbs iterations in which x_t changed, for each t (X has one row per t and one column per sample).
update_rate(X::AbstractMatrix) = [mean(X[t, i] != X[t, i+1] for i in 1:size(X, 2)-1) for t in 1:size(X, 1)]
