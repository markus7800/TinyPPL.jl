# Variational inference on models with a discontinuous density.
#
# In both models below, the observed y depends on the latent x through a branch on x, so the posterior density jumps.
# Reparameterisation gradients (ADVI) are gradients of the log density with respect to x and cannot see the jump,
# score-function gradients (BBVI, ReinforceELBO) only evaluate the log density and can.
# See Lee, Yu, Rival, Yang: Towards Verified Stochastic Variational Inference for Probabilistic Programs (POPL 2020),
# where the second model is taken from.
#
# Run: julia --project=examples examples/universal/discontinuous_vi.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("../common/plotting.jl")

# ===== A step function =====
# y depends on x only through the step function x < 0.5.
# Observing y = 1 rules out x >= 0.5, so the posterior of x is Uniform(0, 0.5).
@ppl function step_model()
    x ~ Uniform(0., 1.)
    y ~ Normal(x < 0.5, 0.1)
    return x
end

args = ()
observations = Observations(:y => 1.)

Random.seed!(0)
lw_retvals, lps = likelihood_weighting(step_model, args, observations, 1_000_000; completion=Evaluation.retval_completion)
W = exp.(lps)
println("Likelihood weighting: P(x < 0.5 | y) ≈ ", sum(W[lw_retvals .< 0.5]))

# ADVI uses reparameterisation gradients, i.e. gradients of the log density with respect to x.
# These are zero almost everywhere for a step function, so ADVI cannot see the observation and fits the prior.
Random.seed!(0)
advi_result = advi_meanfield(step_model, args, observations, 10_000, 10, 0.01)
advi_traces, _ = sample_posterior(advi_result, 100_000)
println("ADVI: P(x < 0.5 | y) ≈ ", mean(retvals(advi_traces) .< 0.5))

# BBVI uses score-function gradients, which only require evaluating the log density, and finds the posterior mass.
Random.seed!(0)
bbvi_result = bbvi(step_model, args, observations, 10_000, 100, 0.01)
bbvi_traces, _ = sample_posterior(bbvi_result, 100_000)
println("BBVI: P(x < 0.5 | y) ≈ ", mean(retvals(bbvi_traces) .< 0.5))

# Instead of the default mean-field guide, we can write the variational family as a guide program with parameters.
# Parameters are initialised to 0 in unconstrained space, so the guide starts at Beta(1, 1), which is the prior.
# The guide's support must lie within the prior's support, otherwise the log joint is -Inf for some samples.
@ppl function step_guide()
    a = param("a", constraint=Positive())
    b = param("b", constraint=Positive())
    x ~ Beta(a, b)
end

# ReinforceELBO estimates the gradient with the score function of the guide, like BBVI.
# (The default MonteCarloELBO relies on reparameterisation gradients and would stay at the prior.)
Random.seed!(0)
guide_result = advi(step_model, args, observations, 10_000, 10, 0.01, step_guide, (), ReinforceELBO())
guide_params = get_constrained_parameters(guide_result.Q)
println("Guide: Beta(", guide_params["a"], ", ", guide_params["b"], ")")
guide_traces, _ = sample_posterior(guide_result, 100_000)
println("Guide: P(x < 0.5 | y) ≈ ", mean(retvals(guide_traces) .< 0.5))

p = histogram(lw_retvals, weights=W, normalize=true, label="LW", linewidth=0, xlabel="x")
histogram!(p, retvals(advi_traces), normalize=true, label="ADVI", linewidth=0, alpha=0.5)
histogram!(p, retvals(bbvi_traces), normalize=true, label="BBVI", linewidth=0, alpha=0.5)
histogram!(p, retvals(guide_traces), normalize=true, label="Beta guide", linewidth=0, alpha=0.5)
show_plot(p, "universal_discontinuous_vi")


# ===== Branching on a latent variable (Lee et al.) =====
# x ~ Normal(0, 5), and y ~ Normal(1, 1) if x > 0, otherwise y ~ Normal(-2, 1). We observe y = 0.
# The posterior is the prior density multiplied by N(0; 1, 1) for x > 0 and by N(0; -2, 1) for x <= 0, so it jumps at 0:
# P(x > 0 | y) = N(0; 1, 1) / (N(0; 1, 1) + N(0; -2, 1)) ≈ 0.818, and since both halves are half-normal with mean ±5 sqrt(2/π),
# E[x | y] = (2 P(x > 0 | y) - 1) * 5 sqrt(2/π) ≈ 2.54.
@ppl function branching()
    x ~ Normal(0., 5.)
    if x > 0
        y ~ Normal(1., 1.)
    else
        y ~ Normal(-2., 1.)
    end
    return x
end

observations = Observations(:y => 0.)
w_pos, w_neg = exp(logpdf(Normal(1., 1.), 0.)), exp(logpdf(Normal(-2., 1.), 0.))
p_pos = w_pos / (w_pos + w_neg)
posterior_density(x) = exp(logpdf(Normal(0., 5.), x)) * (x > 0 ? w_pos : w_neg) / ((w_pos + w_neg) / 2)
println("\nexact: P(x > 0 | y) = ", round(p_pos, digits=3), ", E[x | y] = ", round((2p_pos - 1) * 5 * sqrt(2/π), digits=3))

# Within each branch, the log density is the log prior plus a constant, so the reparameterisation gradient
# is the gradient of the prior and ADVI fits the prior Normal(0, 5).
Random.seed!(0)
advi_result = advi_meanfield(branching, args, observations, 10_000, 10, 0.01)
advi_traces, _ = sample_posterior(advi_result, 100_000)
println("ADVI:  P(x > 0 | y) ≈ ", round(mean(retvals(advi_traces) .> 0), digits=3), ", E[x | y] ≈ ", round(mean(retvals(advi_traces)), digits=3))

# BBVI moves the Gaussian towards x > 0, but a Gaussian cannot represent the jump, so it does not match the posterior exactly.
Random.seed!(0)
bbvi_result = bbvi(branching, args, observations, 10_000, 100, 0.01)
bbvi_traces, _ = sample_posterior(bbvi_result, 100_000)
println("BBVI:  P(x > 0 | y) ≈ ", round(mean(retvals(bbvi_traces) .> 0), digits=3), ", E[x | y] ≈ ", round(mean(retvals(bbvi_traces)), digits=3))

xs = -15:0.05:15
p = histogram(retvals(advi_traces), normalize=true, label="ADVI", linewidth=0, alpha=0.5, xlabel="x")
histogram!(p, retvals(bbvi_traces), normalize=true, label="BBVI", linewidth=0, alpha=0.5)
plot!(p, xs, posterior_density.(xs), label="exact posterior", color=:black)
show_plot(p, "universal_discontinuous_vi_branching")
