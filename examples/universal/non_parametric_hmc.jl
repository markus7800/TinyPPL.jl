# Nonparametric HMC on the pedestrian model (see pedestrian.jl for the model and other inference algorithms).
#
# HMC needs a fixed-dimensional space, but traces of the pedestrian have different numbers of steps.
# Nonparametric HMC (Mak, Zaiser, Ong: Nonparametric Hamiltonian Monte Carlo, ICML 2021) runs HMC on an
# infinite-dimensional trace, of which only the prefix that the program reads is instantiated:
# whenever a leapfrog step makes the program read a new address, its value (and momentum) is sampled and appended.
#
# Run: julia --project=examples examples/universal/non_parametric_hmc.jl

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
n_steps(trace) = count(addr -> addr isa Pair && addr[1] == :step, keys(trace))
print_summary(method, start_positions, steps, W=fill(1 / length(steps), length(steps))) =
    println(rpad(method, 22), "E[start_position | distance] ≈ ", round(start_positions'W, digits=3),
        ", P(number of steps = k | distance) for k = 1..5 ≈ ", [round(sum(W[steps .== k]), digits=3) for k in 1:5])

# Reference posterior.
Random.seed!(0)
lw_traces, logw = likelihood_weighting(pedestrian, args, observations, 1_000_000)
W = exp.(logw)
print_summary("likelihood weighting", retvals(lw_traces), n_steps.(lw_traces.data), W)

# The initial state matters a lot for this model:
# a random prior sample typically walks until distance > 10, which is very unlikely given distance = 1.1.
# HMC cannot escape from there, because decreasing the distance below 10 continues the walk,
# which adds new steps and makes the distance larger again (discontinuity).
# Thus, we start from the most likely of the likelihood weighting samples.
x_initial = lw_traces.data[argmax(logw)]

# non_parametric_hmc(model, args, observations, n_samples, L, eps)
Random.seed!(0)
traces = non_parametric_hmc(pedestrian, args, observations, 50_000, 10, 0.05; x_initial=x_initial)
print_summary("nonparametric HMC", retvals(traces), n_steps.(traces.data))

p = histogram(retvals(traces), normalize=true, label="nonparametric HMC", linewidth=0, alpha=0.6, xlabel="start_position")
density!(p, retvals(lw_traces), weights=W, label="LW reference", color=:black)
show_plot(p, "universal_non_parametric_hmc")
