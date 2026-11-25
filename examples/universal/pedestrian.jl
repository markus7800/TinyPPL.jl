# Introduction to universal models with the pedestrian example.
#
# A universal model (`@ppl function`) is an ordinary Julia function with sample statements.
# Each execution may instantiate a different, even unbounded, set of random variables,
# and branches are evaluated lazily, i.e. only the sample statements of the taken branch are executed.
# Inference algorithms run the program with different samplers, which decide what happens at sample statements.
#
# A pedestrian starts at an unknown position in [0,3] and takes random steps in [-1,1]
# until they arrive home at position 0 (or have walked more than 10 units).
# We observe the total walked distance 1.1 (with noise) and infer the start position.
# The loop has no upper bound, so a trace can contain arbitrarily many steps.
# Example from Mak, Zaiser, Ong: Nonparametric Hamiltonian Monte Carlo (ICML 2021), see also non_parametric_hmc.jl.
#
# Run: julia --project=examples examples/universal/pedestrian.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots
include("../common/plotting.jl")

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
# With many samples, this gives a reference posterior of the start position (takes ~30s).
Random.seed!(0)
lw_retvals, lps = likelihood_weighting(pedestrian, args, observations, 10_000_000; completion=Evaluation.retval_completion)
W = exp.(lps)
println("Likelihood weighting: E[start_position | distance] ≈ ", lw_retvals'W, " (effective sample size ", round(Int, 1 / sum(W.^2)), ")")

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

p = histogram(retvals(lmh_traces), normalize=true, label="LMH", linewidth=0, alpha=0.5, xlabel="start_position")
histogram!(p, retvals(rwmh_traces), normalize=true, label="RWMH", linewidth=0, alpha=0.5)
density!(p, lw_retvals, weights=W, label="LW reference", color=:black)
show_plot(p, "universal_pedestrian_start_position")

# The MCMC traces have different lengths: the posterior over the number of steps.
n_steps = [count(addr -> addr isa Pair && addr[1] == :step, keys(trace)) for trace in rwmh_traces.data]
println("RWMH: P(number of steps = k | distance) for k = 1..6 ≈ ", [round(mean(n_steps .== k), digits=3) for k in 1:6])
p = histogram(n_steps, bins=0.5:1:(maximum(n_steps)+0.5), normalize=true, legend=false, xlabel="number of steps", xticks=1:maximum(n_steps), linewidth=0)
show_plot(p, "universal_pedestrian_steps")
