using TinyPPL.Distributions
using TinyPPL.Graph
import Random

p = Proposal(:x=>Bernoulli(0.), (:x=>:y=>:z)=>Bernoulli(1.));
haskey(p, :x), p[:x]
haskey(p, :x=>:y), p[:x=>:y]
haskey(p, :x=>:y=>:z), p[:x=>:y=>:z]
get(p, :x=>:y=>:z, Bernoulli(0.5))

d = random_walk_proposal_dist(Categorical([0.1, 0.2, 0.7]), 1, 0.5)
rand(d)

b = [true]
model = @ppl Flip begin
    let A = {:a} ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ false),
        C = {:C} ~ Bernoulli(B == 1 ? 0.9 : 0.7)
        
        (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ $(Main.b[1])
        
        A + C
    end
end

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
end

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
retvals'W

@time traces, retvals = lmh(model, 1_000_000);
mean(retvals)

model.logpdf([1., 0., 1., 1.])

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
@time traces, retvals = lmh(model, 1_000_000);

kernels = compile_lmh(model, static_observes=true);
kernels = compile_lmh(model, static_observes=true, proposal=Proposal(:intercept=>Normal(0.,1.)));
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);

mean([r[1] for r in retvals])
mean([r[2] for r in retvals])

xs = [
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
];
ys = [
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
];

model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = $(Main.xs),
        ys = $(Main.ys),
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [{:y=>i} ~ Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i] for i in 1:100]
        
        (slope, intercept)
    end
end;

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000); # 60s
W = exp.(lps);
slope = [r[1] for r in retvals]; slope'W
intercept = [r[2] for r in retvals]; intercept'W

lw = compile_likelihood_weighting(model, static_observes=true)

@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000, static_observes=true); # 1.2s

proposal = Proposal(:slope=>Normal(2.,1.), :intercept=>Normal(-1.,1.));

@time traces, retvals = lmh(model, 1_000_000);
@time traces, retvals = lmh(model, 1_000_000, proposal=proposal); # 60s

kernels = compile_lmh(model, static_observes=true);
kernels = compile_lmh(model, static_observes=true, proposal=proposal);
kernels = compile_lmh(model, [:y], static_observes=true, proposal=proposal);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true); # 2s


mean([r[1] for r in retvals])
mean([r[2] for r in retvals])

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

include("../examples/univariate_gmm/common.jl")

t0 = time_ns()
model = @ppl gmm begin
    function dirichlet(δ, k)
        let w = [{:w=>i} ~ Gamma(δ, 1) for i in 1:k]
            w / sum(w)
        end
    end
    let λ = 3, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
        δ ~ Uniform(5.0-0.5, 5.0+0.5),
        k = 4,
        y = $(Main.gt_ys),
        n = length(y),
        w = dirichlet(δ, k),
        X ~ Categorical(w),
        Y ~ Normal(X, 1.),
        means = [{:μ=>j} ~ Normal(ξ, 1/sqrt(κ)) for j in 1:k],
        vars = [{:σ²=>j} ~ InverseGamma(α, β) for j in 1:k],
        z = [{:z=>i} ~ Categorical(w) for i in 1:n]

        [{:y=>i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]
        
        means
    end
end;
(time_ns() - t0) / 1e9

Random.seed!(0)
X = Vector{Float64}(undef, model.n_variables);
model.sample(X)
model.return_expr(X)
model.logpdf(X) # -3034.9080970776577

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);

@time lw = compile_likelihood_weighting(model)
@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000; static_observes=true); # 42s
W = exp.(lps);

lps[argmax(lps)]
retvals[argmax(lps)]

@time traces, retvals = lmh(model, 1_000_000); # 18s, 90s with full lp computation

@time kernels = compile_lmh(model, static_observes=true);
Random.seed!(0);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);


@time kernels = compile_lmh(model, [:w, :μ, :σ², :z, :y], static_observes=true);
Random.seed!(0);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);


spgm, E = Graph.to_human_readable(model.symbolic_pgm, model.symbolic_return_expr, model.sym_to_ix);


pgm, plates, plated_edges = Graph.plate_transformation(model, [:w, :μ, :σ², :z, :y]);

@ppl obs begin
    let z ~ Bernoulli(0.5),
        μ0 ~ Normal(-1.0, 1.0),
        μ1 ~ Normal(1.0, 1.0),
        y = 0.5

        if z
            Normal(μ0, 1) ↦ y
        else
            Normal(μ1, 1) ↦ y
        end

    end
end


function test()
    x = 0
    for i in [1,2,3,3,2,1,1,2]
        x += i
    end
    x
end
@code_llvm test()

@time test()