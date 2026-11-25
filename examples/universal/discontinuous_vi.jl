# Variational inference on a model with a discontinuous density.
#
# y depends on x only through the step function x < 0.5.
# Observing y = 1 rules out x >= 0.5, so the posterior of x is Uniform(0, 0.5).
# Reparameterisation gradients (ADVI) cannot see the step, score-function gradients (BBVI, ReinforceELBO) can.
#
# Run: julia --project=examples examples/universal/discontinuous_vi.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("../common/plotting.jl")

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
