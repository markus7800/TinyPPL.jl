using TinyPPL.Distributions
using TinyPPL.Traces

function f()
    return 0
end

@ppl function geometric(p::Float64, observed::Bool)
    i = f()
    while true
        b = {(:b,i)} ~ Bernoulli(p)
        b && break
        i += 1
    end
    if observed
        {:X} ~ Normal(i, 1.0)
    end
    return i
end

@macroexpand
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

observations = Dict(:X => 5);
@time traces, retvals, lps = likelihood_weighting(geometric, (0.5, true), observations, 1_000_000);

@time traces, retvals, lps = likelihood_weighting(geometric_recursion, (0.5, true, 0), observations, 1_000_000);

stats = @timed likelihood_weighting(geometric, (0.5, true), observations, 1_000_000);

W = exp.(lps);
P_hat = [sum(W[retvals .== i]) for i in 0:10]

exp(logpdf(Geometric(0.5), 250))
P_X = sum(exp(logpdf(Geometric(0.5), i) + logpdf(Normal(i, 1.0), observations[:X])) for i in 0:250);
P_true = [exp(logpdf(Geometric(0.5), i) + logpdf(Normal(i, 1.0), observations[:X])) / P_X for i in 0:10]

maximum(abs.(P_hat .- P_true))


@ppl function LinReg(xs)
    function f(slope, intercept, x)
        intercept + slope * x
    end

    slope = {:slope} ~ Normal(0.0, 10.)
    intercept = {:intercept} ~ Normal(0.0, 10.)

    for i in 1:length(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), 1.)
    end

    return slope
end

xs = [1., 2., 3., 4., 5.]
ys = [2.1, 3.9, 5.3, 7.7, 10.2]

observations = Dict((:y,i) => ys[i] for i in 1:length(ys));
@time traces, retvals, lps = likelihood_weighting(LinReg, (xs,), observations, 1_000_000);

W = exp.(lps);
retvals'W



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

@ppl function LinReg(xs)
    function f(slope, intercept, x)
        intercept + slope * x
    end

    slope = {:slope} ~ Normal(0.0, 10.)
    intercept = {:intercept} ~ Normal(0.0, 10.)

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

using TinyPPL.Distributions
using TinyPPL.Handlers

function geometric(p::Float64, observed::Union{Nothing,Real}=nothing)
    i = 0
    while true
        b = sample((:b,i), Bernoulli(p))
        b && break
        i += 1
    end
    if !isnothing(observed)
        sample(:X, Normal(i, 1.0), obs=observed)
    end
    return i
end

Handlers.HANDLER_STACK
empty!(Handlers.HANDLER_STACK)

args = (0.3,)
trace_handler = trace(geometric);

import Random
Random.seed!(0)
model_trace = get_trace(trace_handler, args...)


Random.seed!(0)
trace_handler(args...)
trace_handler.trace


replay_handler = trace(replay(geometric, model_trace));
replay_trace = get_trace(replay_handler, args...)


replay_handler = trace(block(replay(geometric, model_trace), msg -> msg["type"] == "observation"));
replay_trace = get_trace(replay_handler, args...)
logpdfsum(replay_trace)

@time traces, retvals, lps = likelihood_weighting(geometric, (0.5, 5.), 1_000_000);


function simple(mean, y)
    X = sample(:X, Normal(mean, 1.))
    sample(:Y, Normal(X, 1.), obs=y)
    return X
end


trace_handler = trace(simple);

trace_handler(0., 1.)
model_trace = trace_handler.trace

replay_handler = trace(block(replay(simple, model_trace), msg -> msg["type"] == "observation"));
replay_handler(0., 1.)
replay_handler.trace



#=
let acc = init
    for i in 1:10
        body(acc, i) 
    end
end
=#