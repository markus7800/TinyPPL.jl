

using TinyPPL.Distributions
using TinyPPL.Evaluation

@ppl function normal()
    X ~ Normal(0., 1.)
    Y ~ Normal(X, 1.)
    Z ~ Normal(Y, 1.)
    return X
end

@ppl function unif()
    X ~ Uniform(-1., 1.)
    Y ~ Uniform(-1. + X, 1. + X)
    Z ~ Uniform(-1. + Y, 1. + Y)
    return X
end

model = normal
model = unif

observations = Dict(:Z => 1.);

@time traces, retvals, lps = likelihood_weighting(model, (), observations, 1_000_000);
W = exp.(lps);
W'retvals
((retvals .- W'retvals).^2)'W
using StatsPlots
histogram(retvals, weights=W, normalize=true);
plot!(x -> exp(logpdf(Normal(1/3,sqrt(2/3)), x)))

@time traces, retvals, lps = lmh(model, (), observations, 1_000_000; proposal=Proposal(), gibbs=false);
@time traces, retvals, lps = rwmh(model, (), observations, 1_000_000);

histogram(retvals, normalize=true)
plot!(x -> exp(logpdf(Normal(1/3,sqrt(2/3)), x)))

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

observations = Dict(:distance => 1.1);

import Random
retvals, lps = begin
    Random.seed!(0)
    N = 10^8
    lps = Vector{Float64}(undef, N)
    retvals = Vector{Float64}(undef, N)
    sampler = LogProb()
    for i in 1:N
        sampler.log_p_Y = 0.
        sampler.log_p_X = 0.
        retvals[i] = pedestrian(Inf, sampler, observations)
        lps[i] = sampler.log_p_Y
    end
    m = maximum(lps)
    l = m + log(sum(exp, lps .- m))
    lps =  lps .- l

    return retvals, lps
end;
W = exp.(lps);
histogram(retvals, weights=W, normalize=true)
density(retvals, weights=W)

@time traces, retvals, lps = lmh(pedestrian, (Inf,), observations, 1_000_000; proposal=Proposal(), gibbs=false);
@time traces, retvals, lps = rwmh(pedestrian, (Inf,), observations, 1_000_000, default_var=0.1);

histogram(retvals, normalize=true)


d = random_walk_proposal_dist(Uniform(-1,1), -0.5, 1.)
histogram([rand(d) for _ in 1:10^6], normalize=true)
plot!(x -> exp(logpdf(d, x)), xlims=(-1.1,1.1))



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


@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000; proposal=Proposal(), gibbs=false);
@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000, proposal=Proposal(:b=>Bernoulli(0.3)), gibbs=false);
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
