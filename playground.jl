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

model.logpdf([1., 0., 1., 1.])

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

model = @ppl LinReg begin
    function get_xs()
        [
            1., 2., 3, 4., 5., 6., 7., 8., 9., 10.,
            11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
            21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
            31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
            41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60.,
            61., 62., 63., 64., 65., 66., 67., 68., 69., 70.,
            71., 72., 73., 74., 75., 76., 77., 78., 79., 80.,
            81., 82., 83., 84., 85., 86., 87., 88., 89., 90.,
            91., 92., 93., 94., 95., 96., 97., 98., 99., 100.
        ]
    end
    function get_ys()
        [
            0.77, 3.94, 5.6, 9.0, 8.95,
            12.17, 11.3, 12.88, 17.56, 18.14,
            20.73, 23.58, 26.16, 27.29, 28.56,
            31.13, 33.17, 33.93, 35.74, 39.2,
            39.93, 41.62, 45.95, 46.71, 49.33,
            50.5, 53.22, 53.34, 58.93, 57.95,
            62.72, 62.33, 65.26, 67.64, 68.72,
            71.02, 71.97, 75.46, 77.28, 79.45,
            80.48, 81.44, 84.04, 88.26, 87.63,
            90.74, 94.25, 94.74, 98.32, 98.9,
            102.86, 103.23, 103.37, 107.81, 109.07,
            111.13, 112.1, 115.76, 117.86, 118.46,
            120.42, 122.09, 123.63, 127.66, 129.04,
            131.88, 134.79, 133.64, 135.52, 139.48,
            141.27, 144.24, 146.11, 148.85, 148.87,
            150.59, 152.62, 154.5, 156.36, 161.28,
            161.85, 162.79, 164.86, 166.88, 169.47,
            170.91, 172.45, 174.53, 177.52, 178.31,
            181.65, 182.27, 184.36, 187.1, 190.0,
            191.64, 193.24, 195.1, 196.71, 199.2
        ]
    end

    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = get_xs(),
        ys = get_ys(),
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:100]
        
        (slope, intercept)
    end
end;

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
slope = [r[1] for r in retvals]; slope'W
intercept = [r[2] for r in retvals]; intercept'W

lw = compile_likelihood_weighting(model)

X = Vector{Float64}(undef, model.n_variables);
@time lw(X)
@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000);



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
model.logpdf(X)

import Tracker
X = Tracker.param(rand(3))
lp = model.logpdf(X)
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