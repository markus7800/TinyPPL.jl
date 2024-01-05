using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using Plots

# UniversalModels may instantiate an arbitary set of random variables per execution.
# There are no assumption about the program structure and this may result in an unbounded number of random variables.
# Branches are executed lazily.

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

Random.seed!(0)
result, lps = likelihood_weighting(pedestrian, args, observations, 5_000_000, (trace, retval) -> retval);
W = exp.(lps);
posterior_mean = result'W

histogram(result, weights=W, normalize=true, legend=false)

Random.seed!(0)
lmh_traces, _ = lmh(pedestrian, args, observations, 1_000_000);
posterior_mean = mean(retvals(lmh_traces))

Random.seed!(0)
rwmh_traces, _ = rwmh(pedestrian, args, observations, 1_000_000, default_var=0.1);
posterior_mean = mean(retvals(rwmh_traces))

Random.seed!(0)
lmh_traces_2, _ = lmh(pedestrian, args, observations, 1_000_000,
    addr2proposal = Addr2Proposal(
        :start_position => ContinuousRandomWalkProposal(0.1, 0., 3.),
        :step => ContinuousRandomWalkProposal(0.1, -1., 1.)
    )
);
posterior_mean = mean(retvals(lmh_traces_2))

# is equivalent
all(rwmh_traces[:start_position] .≈ lmh_traces_2[:start_position])