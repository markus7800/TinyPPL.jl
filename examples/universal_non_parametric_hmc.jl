using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots

@ppl function pedestrian(i_max)
    start_position ~ Uniform(0.,3.)
    position = start_position
    distance = 0.
    i = 1
    while position >= 0. && distance <= 10. && i <= i_max
        step = {:step=>i} ~ Uniform(-1.,1.)
        distance += abs(step)
        position += step
        i += 1
    end
    {:distance} ~ Normal(distance, 0.1)

    return start_position
end

args = (100,)
observations = Observations(:distance => 1.1);

# reference posterior of start_position
Random.seed!(0); lw_retvals, lps = likelihood_weighting(pedestrian, args, observations, 1_000_000, Evaluation.retval_completion);
W = exp.(lps .- maximum(lps)); W ./= sum(W);

# The initial state matters a lot for this model:
# a random prior sample typically walks until distance > 10, which is very unlikely given distance = 1.1.
# HMC cannot escape from there, because decreasing the distance below 10 continues the walk,
# which adds new steps and makes the distance larger again (discontinuity).
# Thus, we start from the most likely trace out of 1000 prior samples.
Random.seed!(0); traces, lw = likelihood_weighting(pedestrian, args, observations, 1_000);
x_initial = traces.data[argmax(lw)]

Random.seed!(0); @time result = non_parametric_hmc(pedestrian, args, observations, 10_000, 10, 0.05; x_initial=x_initial);

# result contains constrained values and return values
start_position = retvals(result)

histogram(start_position, normalize=true, legend=false)
density!(lw_retvals, weights=W)
