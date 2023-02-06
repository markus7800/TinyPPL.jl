using TinyPPL.Distributions
using TinyPPL.Traces

@ppl function geometric(p::Float64, observed::Bool)
    i = 0
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


using TinyPPL.Graph

model = @ppl Flip begin
    let A ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ false),
        C ~ Bernoulli(B == 1 ? 0.9 : 0.7),
        D = (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ true
        
        A + C
    end
end;


model = @ppl Flip2 begin
    function plus(x, y)
        let a = 1, b = 1
            x + y + a - b
        end
    end
    let A ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ false),
        C ~ Bernoulli(B == 1 ? 0.9 : 0.7),
        D = (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ true
        
        plus(A, C)
    end
end;

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);

W = exp.(lps);
retvals'W

model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = [1., 2., 3., 4., 5.],
        ys = [2.1, 3.9, 5.3, 7.7, 10.2],
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:5]
        
        (slope, intercept)
    end
end

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
slope = [r[1] for r in retvals]; slope'W
intercept = [r[2] for r in retvals]; intercept'W



@time traces, retvals, lps = hmc(model, 10_000, 0.05, 10, [1. 0.; 0. 1.]);
mean(retvals)

model = @ppl simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end

model = @ppl NormalChain begin
    let x ~ Normal(0., 1.),
        y ~ Normal(x, 1.),
        z = (Normal(y, 1.) ↦ 0)
        x
    end
end

X = Vector{Float64}(undef, model.n_variables);
model.log_pdf(X)

import Tracker
X = Tracker.param(rand(3))
lp = model.log_pdf(X)
Tracker.back!(lp);
Tracker.grad(X)
X_data = Tracker.data(X)

using TinyPPL.Distributions
function compare_lp(y, x, z)
    return logpdf(Normal(0., 1.), x) + logpdf(Normal(x, 1.), y) + logpdf(Normal(y, 1.), z)
end

X_tracked = Tracker.param.(X_data)
lp_2 = compare_lp(X_tracked...)
Tracker.back!(lp_2)
Tracker.grad.(X_tracked)

@time hmc(model, 1_000, 0.05, 10, [1. 0.; 0. 1.]);

using TinyPPL.Distributions
using TinyPPL.Evaluation

@ppl function geometric(p::Float64, observed::Bool)
    i = 0
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

observations = Dict(:X => 5);
sampler = Forward();
@time [geometric(0.5, true, sampler, observations) for _ in 1:5_000_000];

@time traces, retvals, lps = likelihood_weighting(geometric, (0.5, true), observations, 1_000_000);

W = exp.(lps);
P_hat = [sum(W[retvals .== i]) for i in 0:10]



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

replay_handler = trace(replay(geometric, model_trace));
replay_trace = get_trace(replay_handler, args...)


replay_handler = trace(block(replay(geometric, model_trace), msg -> msg["type"] == "observation"));
replay_trace = get_trace(replay_handler, args...)
logpdfsum(replay_trace)

@time traces, retvals, lps = likelihood_weighting(geometric, (0.5, 5.), 1_000_000);


@macroexpand @ppl function simple(mean::Float64)
    X = {:X} ~ Normal(mean, 1.)
    {:Y} ~ Normal(X, 1.)
    return X
end