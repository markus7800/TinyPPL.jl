
using TinyPPL.Distributions
using TinyPPL.Evaluation
using TinyPPL.Graph
import Random
using Plots


@ppl function gap_model(size)
    x ~ Uniform(-1,1)
    if x <= 0
        y ~ Uniform(0, size)
    end
end
size = 2.
args = (size,)
observations = Observations()

logjoint = Evaluation.make_logjoint(gap_model, args, observations)
logjoint(UniversalTrace(:x => 1., :y => -100.))

xs = -1:0.1:1
ys = 0.:0.1:size
ps = [exp(logjoint(UniversalTrace(:x => x, :y => y))) for y in ys, x in xs]

heatmap(xs, ys, ps)

Random.seed!(0); traces = Evaluation.rwmh(gap_model, args, observations, 10^6);
histogram(traces[:x], legend=false, normalize=true)


@ppl static function static_gap_model(size)
    x ~ Uniform(-1,1)
    if x <= 0
        y ~ Uniform(0, size)
    end
end

logjoint, addresse_to_ixs = Evaluation.make_logjoint(static_gap_model, args, observations)

Random.seed!(0); traces = Evaluation.hmc(static_gap_model, args, observations, 1_000_000, 1, 0.01, ad_backend=:forwarddiff, unconstrained=true)
histogram(traces[:x], legend=false, normalize=true)

Random.seed!(0); traces = Evaluation.hmc(static_gap_model, args, observations, 100_000, 10, 0.01, ad_backend=:forwarddiff, unconstrained=true)
histogram(traces[:x], legend=false, normalize=true)

Random.seed!(0); traces = Evaluation.hmc(static_gap_model, args, observations, 1_000_000, 1, 0.01, ad_backend=:forwarddiff, unconstrained=false)
histogram(traces[:x], legend=false, normalize=true)

# Towards Verified Stochastic Variational Inference for Probabilistic Programs
@ppl static function branching()
    x ~ Normal(0.,5.)
    if x > 0
        y ~ Normal(1.,1.)
    else
        y ~ Normal(-2.,1.)
    end
end
args = ()
observations = Observations(:y => 0.)

logjoint, addresse_to_ixs = Evaluation.make_logjoint(branching, args, observations)

xs = -15:0.1:15
ps = [exp(logjoint([x])) for x in xs]
ps = ps / (0.1*sum(ps))

plot(xs, ps, legend=false)

Random.seed!(0); traces = Evaluation.rwmh(to_universal(branching), args, observations, 10^6);
histogram(traces[:x], legend=false, normalize=true)


Random.seed!(0); vi_result = Evaluation.bbvi(branching, args, observations, 10_000, 100, 0.01; ad_backend=:forwarddiff);
traces = Evaluation.sample_posterior(vi_result, 10^6)
histogram(traces[:x], legend=false, normalize=true)

Random.seed!(0); vi_result = Evaluation.advi_meanfield(branching, args, observations, 10_000, 10, 0.01);
traces = Evaluation.sample_posterior(vi_result, 10^6)
histogram(traces[:x], legend=false, normalize=true)

@ppl static function branching_guide()
    θ = param("theta")
    x ~ Normal(θ,1)
end
guide_args = ()

Random.seed!(0); vi_result = Evaluation.advi(
    branching, args, observations,
    100_000, 10, 0.01,
    branching_guide, guide_args, ReinforceELBO());

traces = Evaluation.sample_posterior(vi_result, 10^6)

plot(xs, ps, legend=false)
histogram!(traces[:x], legend=false, normalize=true)

mean(traces[:x])