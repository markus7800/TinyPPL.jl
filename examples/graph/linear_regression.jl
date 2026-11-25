# Bayesian linear regression with all sampling-based inference algorithms for graph-based models.
#
# The posterior is Gaussian and known exactly (see common/linear_regression.jl),
# so every algorithm can be checked against it. The same model as static model: static/linear_regression.jl.
#
# Run: julia --project=examples examples/graph/linear_regression.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import Random
include("../common/linear_regression.jl")

model = @pgm linear_regression begin
    let xs = $(xs),
        ys = $(ys),
        intercept ~ Normal(0., $(prior_σ)),
        slope ~ Normal(0., $(prior_σ))

        [({:y => i} ~ Normal(intercept + slope * xs[i], $(σ)) ↦ ys[i]) for i in 1:$(length(xs))]

        (intercept, slope)
    end
end

print_posterior("exact", posterior_μ, posterior_std)

# Importance sampling from the prior.
Random.seed!(0)
traces, logw, log_Z = likelihood_weighting(model, 1_000_000)
print_posterior("likelihood weighting", traces, exp.(logw))

# Single-site MCMC: LMH proposes from the prior, RWMH proposes close to the current value.
Random.seed!(0)
traces = lmh(model, 1_000_000)
print_posterior("LMH", traces)

Random.seed!(0)
traces = rwmh(model, 1_000_000)
print_posterior("RWMH", traces)

# HMC uses the gradient of the log joint and updates all variables jointly,
# so far fewer samples are needed for the correlated posterior: hmc(model, n_samples, L, eps).
Random.seed!(0)
traces = hmc(model, 10_000, 10, 0.1)
print_posterior("HMC", traces)

# Sequential Monte Carlo resamples the particles after each observation.
# Nodes are visited in topological order, so no Libtask is needed as for evaluation-based models.
# `light_smc` is a minimal implementation that resamples at every observation.
Random.seed!(0)
traces, logw, log_Z_smc = smc(model, 100_000)
print_posterior("SMC", traces, exp.(logw))

Random.seed!(0)
traces, logw, _ = light_smc(model, 100_000)
print_posterior("light SMC", traces, exp.(logw))

# Particle independent Metropolis-Hastings uses a full SMC run (here with 100 particles) as proposal
# and accepts based on the SMC evidence estimates. The acceptance rate increases with more particles.
Random.seed!(0)
traces = particle_IMH(model, 100, 1_000)
print_posterior("particle IMH", traces)

println("\nlog p(y): exact = ", round(log_evidence, digits=3),
    ", likelihood weighting ≈ ", round(log_Z, digits=3), ", SMC ≈ ", round(log_Z_smc, digits=3))
