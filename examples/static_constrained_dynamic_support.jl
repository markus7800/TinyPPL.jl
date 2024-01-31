using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using Plots

@ppl static function unif()
    x ~ Uniform(-1,1)
    y ~ Uniform(x-1,x+1)
    z ~ Uniform(y-1,y+1)
end
args = ()
observations = Observations()

Random.seed!(0)
@time vi_result = advi_meanfield(unif, args, observations,  10_000, 10, 0.01)
posterior = sample_posterior(vi_result, 1_000_000);
histogram(posterior[:y], normalize=true, legend=false)

Random.seed!(0)
@time vi_result = bbvi(unif, args, observations,  10_000, 10, 0.01)
posterior = sample_posterior(vi_result, 1_000_000);
histogram(posterior[:y], normalize=true, legend=false)

Random.seed!(0)
@time traces = hmc(unif, args, observations, 100_000, 10, 0.1; unconstrained=true, ad_backend=:forwarddiff);
histogram(traces[:y], normalize=true, legend=false)

# unconstrained=false has poor results:
Random.seed!(0)
@time traces = hmc(unif, args, observations, 100_000, 10, 0.1; unconstrained=false, ad_backend=:forwarddiff);
histogram(traces[:y], normalize=true, legend=false)