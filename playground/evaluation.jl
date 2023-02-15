

using TinyPPL.Distributions
using TinyPPL.Evaluation

@ppl function geometric(p::Float64, observed::Bool)
    i = 0
    while true
        b = {:b => i} ~ Bernoulli(p)
        b && break
        i += 1
    end
    if observed
        {:X} ~ Normal(i, 1.0)
    end
    return i
end

observations = Dict(:X => 5);
sampler = Forward();
@time [geometric(0.5, true, sampler, observations) for _ in 1:5_000_000];

@time traces, retvals, lps = likelihood_weighting(geometric, (0.5, true), observations, 1_000_000);

W = exp.(lps);
P_hat = [sum(W[retvals .== i]) for i in 0:10]


@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000, Proposal());
@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000, Proposal(:b=>Bernoulli(0.3)));
@time traces, retvals, lps = rwmh(geometric, (0.5, true), observations, 1_000_000);

P_hat = [mean(retvals .== i) for i in 0:10]

function f(slope, intercept, x)
    intercept + slope * x
end

@ppl function LinReg(xs)
    slope = {:slope} ~ Normal(0.0, 10.)
    intercept ~ Normal(0.0, 10.)

    for i in 1:length(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), 1.)
    end

    return (slope, intercept)
end

xs = [1., 2., 3., 4., 5.]
ys = [2.1, 3.9, 5.3, 7.7, 10.2]

observations = Dict((:y,i) => ys[i] for i in 1:length(ys));
@time traces, retvals, lps = likelihood_weighting(LinReg, (xs,), observations, 1_000_000);
W = exp.(lps);
[r[1] for r in retvals]'W
[r[2] for r in retvals]'W

@time traces, retvals, lps = lmh(LinReg, (xs,), observations, 1_000_000);
@time traces, retvals, lps = rwmh(LinReg, (xs,), observations, 1_000_000, default_var=0.01);
mean(r[1] for r in retvals)
mean(r[2] for r in retvals)


using MacroTools

# MacroTools.@capture(:(x ~ Normal(0, 1)), var_ ~ dist_)

MacroTools.@capture(:({:x} ~ Normal(0, 1)), var_ ~ dist_)
