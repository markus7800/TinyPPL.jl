using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using Plots

@ppl static function normal(N)
    X = {:X} ~ Normal(0., 1.)
    for i in 1:N
        X = {:X => i} ~ Normal(0., 1.)
    end
    Z ~ Normal(X, 1.)
    return X
end
observations = Observations(:Z => 1.);

Random.seed!(0)
@time traces, lps = likelihood_weighting(normal, (100,), observations, 1_000_000);
@time traces, lps = likelihood_weighting(normal, (100,), observations, 1_000_000, Address[:X]);