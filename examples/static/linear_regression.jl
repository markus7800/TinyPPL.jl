# Bayesian linear regression with all sampling-based inference algorithms for static models.
#
# The posterior is Gaussian and known exactly (see common/linear_regression.jl),
# so every algorithm can be checked against it. Variational inference on the same model: vi.jl.
# The same model as graph-based model: graph/linear_regression.jl.
#
# Run: julia --project=examples examples/static/linear_regression.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("../common/linear_regression.jl")

@ppl static function linear_regression(xs)
    intercept ~ Normal(0., prior_σ)
    slope ~ Normal(0., prior_σ)
    for i in eachindex(xs)
        {:y => i} ~ Normal(intercept + slope * xs[i], σ)
    end
    return (intercept, slope)
end

args = (xs,)
observations = Observations((:y => i) => y for (i, y) in enumerate(ys))

print_posterior("exact", posterior_μ, posterior_std)

# Importance sampling from the prior.
Random.seed!(0)
traces, logw, log_Z = likelihood_weighting(linear_regression, args, observations, 1_000_000)
print_posterior("likelihood weighting", traces, exp.(logw))

# Single-site MCMC is implemented for universal models only.
# Every static model is also a universal model, `to_universal` converts it.
Random.seed!(0)
traces = lmh(to_universal(linear_regression), args, observations, 1_000_000)
print_posterior("LMH", traces)

Random.seed!(0)
traces = rwmh(to_universal(linear_regression), args, observations, 1_000_000)
print_posterior("RWMH", traces)

# HMC uses the gradient of the log joint and updates all variables jointly,
# so far fewer samples are needed for the correlated posterior: hmc(model, args, observations, n_samples, L, eps).
Random.seed!(0)
traces = hmc(linear_regression, args, observations, 10_000, 10, 0.1)
print_posterior("HMC", traces)

# Sequential Monte Carlo resamples the particles after each observation.
Random.seed!(0)
traces, logw, log_Z_smc = smc(linear_regression, args, observations, 10_000)
print_posterior("SMC", traces, exp.(logw))

# Particle independent Metropolis-Hastings uses a full SMC run (here with 100 particles) as proposal
# and accepts based on the SMC evidence estimates. The acceptance rate increases with more particles.
Random.seed!(0)
traces = particle_IMH(linear_regression, args, observations, 100, 200)
print_posterior("particle IMH", traces)

println("\nlog p(y): exact = ", round(log_evidence, digits=3),
    ", likelihood weighting ≈ ", round(log_Z, digits=3), ", SMC ≈ ", round(log_Z_smc, digits=3))
