
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

function get_true_posterior(p, n, X)
    P_X = sum(exp(logpdf(Geometric(p), i) + logpdf(Normal(i, 1.0), X)) for i in 0:250);
    P_true = [exp(logpdf(Geometric(p), i) + logpdf(Normal(i, 1.0), X)) / P_X for i in 0:n]
    return P_true
end


@ppl function geometric(p::Float64)
    i = 0
    while true
        b = {(:b,i)} ~ Bernoulli(p)
        b && break
        i += 1
    end
    {:X} ~ Normal(i, 1.0)
    return i
end


@ppl function geometric_recursion(p::Float64, observed::Bool, i::Int)
    b = {(:b,i)} ~ Bernoulli(p)
    if b
        if observed
            {:X} ~ Normal(i, 1.0)
        end
        return i
    else
        return @subppl geometric_recursion(p, observed, i+1)
    end
end

p = 0.5;
args = (p,)
observations = Observations(:X => 5);
posterior_true = get_true_posterior(p, 10, observations[:X])

@info "likelihood_weighting"
_ = likelihood_weighting(geometric, args, observations, 1_000); # for compilation
Random.seed!(0); @time traces, lps = likelihood_weighting(geometric, args, observations, 1_000_000);

W = exp.(lps);
posterior_est = [sum(W[traces.retvals .== i]) for i in 0:10]
posterior_diff = maximum(abs.(posterior_true .- posterior_est))


@time traces, lps = likelihood_weighting(geometric_recursion, (p, true, 0), observations, 1_000_000);
W = exp.(lps);
posterior_est = [sum(W[traces.retvals .== i]) for i in 0:10]
posterior_diff = maximum(abs.(posterior_true .- posterior_est))