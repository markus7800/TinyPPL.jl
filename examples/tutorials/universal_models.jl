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

args = (100,)
observations = Observations(:distance => 1.1);

Random.seed!(0)
lw_result, lps = likelihood_weighting(pedestrian, args, observations, 5_000_000, (trace, retval) -> retval);
W = exp.(lps);
posterior_mean = lw_result'W

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
@assert all(rwmh_traces[:start_position] .≈ lmh_traces_2[:start_position])


histogram(lw_result, weights=W, normalize=true, legend=false)
histogram!(retvals(lmh_traces), normalize=true, linewidth=0, alpha=0.5)
histogram!(retvals(rwmh_traces), normalize=true, linewidth=0, alpha=0.5)








Random.seed!(0)
vi_result = advi_meanfield(pedestrian, args, observations, 1_000, 10, 0.1);
length(vi_result.Q.variational_dists)
traces, lps = sample_posterior(vi_result, 100_000);
mean(retvals(traces))

histogram(retvals(traces), normalize=true, legend=false)



@ppl function unif()
    x ~ Uniform(0.,1.)
    y ~ Normal(x < 0.5, 0.1)
    x
end
args = ()
observations = Observations(:y => 1.)

Random.seed!(0)
lw_result, lps = likelihood_weighting(unif, args, observations, 1_000_000, (trace, retval) -> retval);
W = exp.(lps);

histogram(lw_result, weights=W, normalize=true, legend=false)

Random.seed!(0)
vi_result = advi_meanfield(unif, args, observations, 10_000, 10, 0.01);
traces, lps = sample_posterior(vi_result, 100_000);
mean(retvals(traces))
histogram(retvals(traces), normalize=true, legend=false)


@ppl function unif_guide()
    mu = param("mu")
    sigma = param("sigma", constraint=Positive())
    a = param("a", constraint=ZeroToOne())

    # a = param("a", constraint=Positive())
    # b = param("b", constraint=Positive())
 
    T = transform_to(RealInterval(0, a))
    x ~ TransformedDistribution(Normal(mu, sigma), T)
    # x ~ Beta(a, b)
end

Random.seed!(0)
vi_result = advi(unif, args, observations, 10_000, 10, 0.1, unif_guide, (), MonteCarloELBO());
p = get_constrained_parameters(vi_result.Q)
p["a"]
p["b"]

traces, lps = sample_posterior(vi_result, 100_000);
histogram(retvals(traces), normalize=true, legend=false)

X, lp = Evaluation.rand_and_logpdf(vi_result.Q)