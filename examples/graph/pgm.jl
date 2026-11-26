# Introduction to graph-based models.
#
# `@pgm` compiles a restricted modelling language (let blocks, static loops, first-order functions without recursion)
# into a probabilistic graphical model with a fixed set of nodes. Each node has a distribution that depends only on
# its parents, which gives inference algorithms full control over the model structure.
# See the readme for the full syntax.
#
# Run: julia --project=examples examples/graph/pgm.jl

using TinyPPL.Distributions
using TinyPPL.Graph
# only for the comparison with an evaluation-based model (both modules export e.g. likelihood_weighting)
using TinyPPL.Evaluation: @ppl, Observations, UniversalTrace, make_logjoint
import Random

# ===== A first model =====
# `X ~ dist` samples a latent variable, `dist ↦ value` observes a value.
# The printed model shows the nodes (latent xi and observed yi), their distributions, dependencies and return expression.
model = @pgm simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end
println(model)

# The posterior of X is Normal(0.5, sqrt(0.5)).
Random.seed!(0)
traces, logw, log_Z = likelihood_weighting(model, 100_000)
println("Likelihood weighting: E[X | Y] ≈ ", traces[:X]'exp.(logw), " (exact: 0.5)\n")


# ===== Functions, data and plates =====
# Functions are defined in the model block and inlined. Data is interpolated with $(...),
# which is evaluated where @pgm is called, so local variables are visible.
# Static generators [... for i in 1:n] are unrolled, here into one observed node per data point.
# The `plated` annotation groups nodes with addresses :y => i into a plate, which speeds up compilation for large models.
xs = [-1., -0.5, 0.0, 0.5, 1.0]
ys = [-3.2, -1.8, -0.5, -0.2, 1.5]

model = @pgm plated linear_regression begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = $(xs),
        ys = $(ys),
        slope ~ Normal(0., 3.),
        intercept ~ Normal(0., 3.)

        [({:y => i} ~ Normal(f(slope, intercept, xs[i]), 2.) ↦ ys[i]) for i in 1:5]

        (slope, intercept)
    end
end
println(model)
println("Plates: ", model.plate_info.plates)
println("Plated edges: ", sort(collect(model.plate_info.plated_edges), by=string), "\n")


# ===== Loops =====
# @iterate(n, f, x) unrolls f(f(...f(x))) and @loop(n, f, x, args...) unrolls x = f(i, x, args...) for i in 1:n.
model = @pgm iterate begin
    function f(x)
        x + 1
    end
    let X ~ Uniform(-1., 1.)
        @iterate(3, f, X)
    end
end
println(model)

# A linear Gaussian state space model with three time steps:
# each step samples the next state and observes it with noise.
y = [0.5, -0.2, 0.8]
model = @pgm lgss begin
    function step(t, x, a, σ_v, σ_e, y)
        let new_x = {:x => t} ~ Normal(a * x, σ_v)
            {:y => t} ~ Normal(new_x, σ_e) ↦ y[t]
            new_x
        end
    end
    let y = $(y)
        @loop(3, step, 0., 0.9, 0.32, 1., y)
    end
end
println(model)


# ===== Semantics of branching =====
# By default, both branches of an if statement are part of the graph and all nodes are always scored (eager semantics).
# The node x contributes to the density even if b = 0.
model = @pgm eager_branching begin
    let b ~ Bernoulli(0.5)
        if b == 1.
            let x ~ Normal(-1, 1)
                x
            end
        end
    end
end
println(
    "Eager: log p(b=0, x=-10) = ", model.logpdf([0., -10.], Float64[]),
    ", log p(b=1, x=-10) = ", model.logpdf([1., -10.], Float64[]))

# With `lazy_ifs`, nodes in the branch not taken get a Flat distribution (density 1, sampled as NaN),
# which matches the semantics of evaluation-based models.
model = @pgm lazy_ifs lazy_branching begin
    let b ~ Bernoulli(0.5)
        if b == 1.
            let x ~ Normal(-1, 1)
                x
            end
        end
    end
end
println(
    "Lazy: log p(b=0) = ", model.logpdf([0., NaN], Float64[]),
    ", log p(b=1, x=-10) = ", model.logpdf([1., -10.], Float64[]))

@ppl function universal_branching()
    b ~ Bernoulli(0.5)
    if b == 1
        x ~ Normal(-1, 1)
    end
end
logjoint = make_logjoint(universal_branching, (), Observations())
println(
    "Universal: log p(b=0) = ", logjoint(UniversalTrace(:b => 0.)),
    ", log p(b=1, x=-10) = ", logjoint(UniversalTrace(:b => 1., :x => -10.)), "\n")


# ===== Calling functions defined outside the model =====
# Only base Julia functions and distributions are available in the model, other functions have to be qualified with
# their module. get_retval evaluates the return expression for given values of the latent variables.
f_main(x) = x^2
model = @pgm function_calls begin
    function f_pgm(x)
        x^3
    end
    let x ~ Normal(0., 1)
        (Main.f_main(x), f_pgm(x))
    end
end
println(model)
println("Return value at x = 2: ", get_retval(model, [2.]))
