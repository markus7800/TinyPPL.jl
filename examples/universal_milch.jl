

# Milch et. al. Approximate Inference for Infinite Contingent Bayesian Networks

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

# noise free
@ppl function urn(K::Int)
    N ~ Poisson(6)
    balls = []
    for i in 1:N
        ball = {:ball => i} ~ Bernoulli(0.5)
        push!(balls, ball)
    end
    n_black = 0
    if N > 0
        for k in 1:K
            ball_ix = {:drawn_ball => k} ~ DiscreteUniform(1,N)
            n_black += balls[ball_ix]
        end
    end
    {:n_black} ~ Dirac(n_black)
    return N
end

args = (10,)
observations = Observations(:n_black => 5);

# Random.seed!(0); @time traces, lps = likelihood_weighting(urn, args, observations, 5_000_000);
Random.seed!(0); @time retvals, lps = likelihood_weighting(urn, args, observations, 5_000_000, Evaluation.retval_completion);
Ns = retvals;
W = exp.(lps);
[sum(W[Ns .== n]) for n in 1:15]
