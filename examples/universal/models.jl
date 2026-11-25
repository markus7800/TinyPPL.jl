# Introduction to universal models.
#
# A universal model (`@ppl function`) is an ordinary Julia function with sample statements.
# Each execution may instantiate a different, even unbounded, set of random variables,
# and branches are evaluated lazily, i.e. only the sample statements of the taken branch are executed.
# Inference algorithms run the program with different samplers, which decide what happens at sample statements.
#
# Run: julia --project=examples examples/universal/models.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("../common/plotting.jl")

# ===== Pedestrian =====
# A pedestrian starts at an unknown position in [0,3] and takes random steps in [-1,1]
# until they arrive home at position 0 (or have walked more than 10 units).
# We observe the total walked distance 1.1 (with noise) and infer the start position.
# The number of steps, and thus of random variables, is random.

@ppl function pedestrian()
    start_position ~ Uniform(0., 3.)
    position = start_position
    distance = 0.
    i = 1
    while position >= 0. && distance <= 10.
        step = {:step => i} ~ Uniform(-1., 1.)
        distance += abs(step)
        position += step
        i += 1
    end
    {:distance} ~ Normal(distance, 0.1)

    return start_position
end

args = ()
observations = Observations(:distance => 1.1)

# Likelihood weighting samples from the prior and weights each sample by the likelihood of the observations.
# The log weights are normalised, i.e. sum(exp.(lps)) == 1.
# `retval_completion` stores only the return values instead of full traces to save memory.
Random.seed!(0)
lw_retvals, lps = likelihood_weighting(pedestrian, args, observations, 1_000_000; completion=Evaluation.retval_completion)
W = exp.(lps)
println("Likelihood weighting: E[start_position | distance] ≈ ", lw_retvals'W)

# Lightweight Metropolis-Hastings resamples a single random variable from its prior per step and re-runs the program.
# Random variables at new addresses (e.g. additional steps) are sampled from the prior.
Random.seed!(0)
lmh_traces = lmh(pedestrian, args, observations, 1_000_000)
println("LMH: E[start_position | distance] ≈ ", mean(retvals(lmh_traces)))

# Random-walk Metropolis-Hastings proposes a new value close to the current value instead.
Random.seed!(0)
rwmh_traces = rwmh(pedestrian, args, observations, 1_000_000, default_var=0.1)
println("RWMH: E[start_position | distance] ≈ ", mean(retvals(rwmh_traces)))

# LMH accepts custom proposals per address (an address `:step` applies to all addresses `:step => i`).
# With random-walk proposals it is the same algorithm as RWMH and produces the same chain.
Random.seed!(0)
lmh_rw_traces = lmh(pedestrian, args, observations, 1_000_000,
    addr2proposal = Addr2Proposal(
        :start_position => ContinuousRandomWalkProposal(0.1, 0., 3.),
        :step => ContinuousRandomWalkProposal(0.1, -1., 1.)
    )
)
println("LMH with random-walk proposals equals RWMH: ", all(rwmh_traces[:start_position] .≈ lmh_rw_traces[:start_position]))

p = histogram(lw_retvals, weights=W, normalize=true, label="LW", linewidth=0, xlabel="start_position")
histogram!(p, retvals(lmh_traces), normalize=true, label="LMH", linewidth=0, alpha=0.5)
histogram!(p, retvals(rwmh_traces), normalize=true, label="RWMH", linewidth=0, alpha=0.5)
show_plot(p, "universal_models_pedestrian")


# ===== Variational inference on a discontinuous model =====
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
show_plot(p, "universal_models_step")
