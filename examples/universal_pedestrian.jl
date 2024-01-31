
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

args = (Inf,)
observations = Observations(:distance => 1.1);


Random.seed!(0); lw_retvals, lps = likelihood_weighting(pedestrian, args, observations, 5_000_000, Evaluation.retval_completion);
W = exp.(lps);

density(lw_retvals, weights=W, legend=false)

Random.seed!(0); @time traces = lmh(pedestrian, args, observations, 1_000_000; gibbs=false);

Random.seed!(0); @time traces = rwmh(pedestrian, args, observations, 1_000_000, default_var=0.1);

histogram(retvals(traces), normalize=true, legend=false)
density!(lw_retvals, weights=W)
