using TinyTinyPPL.Distributions
using TinyPPL.TraceBased

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


W = exp.(lps);
P_hat = [sum(W[retvals .== i]) for i in 0:10]

P_X = sum(exp(logpdf(Geometric(0.5), i) + logpdf(Normal(i, 1.0), observations[:X])) for i in 0:100);
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

model = @pgm Flip begin
    let A ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ false),
        C ~ Bernoulli(B == 1 ? 0.9 : 0.7),
        D = (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ true
        
        A + C
    end
end;


model = @pgm Flip2 begin
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

model = @pgm LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = [1., 2., 3., 4., 5.],
        ys = [2.1, 3.9, 5.3, 7.7, 10.2],
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:5]
        slope
        # [ys[i] for i in 1:5]
    end
end

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);

W = exp.(lps);
retvals'W

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

