module GraphTests
# Graph-based (PGM) approximate inference (src/graph, without src/graph/exact).

using Test, Random, Statistics
using TinyPPL.Distributions
using TinyPPL.Graph
import TinyPPL.Graph as G
import Distributions: pdf
import Tracker, ForwardDiff, ReverseDiff
import ..References as R
import TinyPPL
using ..TestUtils

# ---------------------------------------------------------------------- models
# see references.jl for the mathematical definitions.
# Data is mostly inlined, but @pgm can also use variables of the calling module (see "models in other modules").

const normal_normal = @pgm graph_test_normal_normal begin
    let μ ~ Normal(0., 2.),
        ys = [1.0, 2.0, 1.5]
        [({:y => i} ~ Normal(μ, 1.) ↦ ys[i]) for i in 1:3]
        μ
    end
end

# normal_normal with latent after last observe (C14)
const normal_normal_trailing = @pgm graph_test_normal_normal_trailing begin
    let μ ~ Normal(0., 2.),
        ys = [1.0, 2.0, 1.5]
        [({:y => i} ~ Normal(μ, 1.) ↦ ys[i]) for i in 1:3]
        let w ~ Normal(μ, 1.)
            w
        end
    end
end

const gamma_scale = @pgm graph_test_gamma_scale begin
    let s ~ Gamma(2., 1.),
        ys = [0.5, -1.2, 2.0]
        [({:y => i} ~ Normal(0., s) ↦ ys[i]) for i in 1:3]
        s
    end
end

# constrained latents that are parents of constrained latents (C5)
const hierarchical_scale = @pgm graph_test_hierarchical_scale begin
    let s ~ Gamma(2., 1.),
        b ~ Uniform(0., s),
        x ~ Normal(b, s)
        {:y} ~ Normal(x, 1.) ↦ 0.3
        x
    end
end

# observation y2 depends on past (z) and future (a) latent when resampling at y1 (C13)
const future_parents = @pgm graph_test_future_parents begin
    let z ~ Normal(0., 1.)
        {:y1} ~ Normal(z, 1.) ↦ 1.
        let a ~ Normal(z, 1.)
            {:y2} ~ Normal(a + z, 1.) ↦ 2.5
            z
        end
    end
end
# linear Gaussian posterior of (z, a)
const FP_PRIOR_COV = [1. 1.; 1. 2.]
const FP_H = [1. 0.; 1. 1.]
const FP_POST_COV = inv(inv(FP_PRIOR_COV) + FP_H' * FP_H)
const FP_POST_MEAN = FP_POST_COV * FP_H' * [1., 2.5]

const beta_bernoulli = @pgm graph_test_beta_bernoulli begin
    let p ~ Beta(2., 2.),
        ys = [1., 1., 0., 1., 1.]
        [({:y => i} ~ Bernoulli(p) ↦ ys[i]) for i in 1:5]
        p
    end
end

const discrete_network = @pgm graph_test_discrete_network begin
    let z ~ Categorical([0.5, 0.3, 0.2]),
        b ~ Bernoulli(z == 1. ? 0.1 : (z == 2. ? 0.5 : 0.9))
        {:o1} ~ Bernoulli(b == 1. ? 0.8 : 0.3) ↦ 1.
        {:o2} ~ Bernoulli(z == 3. ? 0.7 : 0.2) ↦ 0.
        z
    end
end

const mixed = @pgm graph_test_mixed begin
    let z ~ Bernoulli(0.3),
        x ~ Normal(z == 1. ? 2.0 : -1.0, 1.0)
        {:y} ~ Normal(x, 0.5) ↦ 0.8
        x
    end
end

# address order (:a, :o, :z) differs from topological order (z, a, o)
const chain = @pgm graph_test_chain begin
    let z ~ Normal(0., 1.),
        a ~ Normal(z, 0.5)
        {:o} ~ Normal(a, 1.) ↦ 1.
        z
    end
end

const linear_regression = @pgm graph_test_linear_regression begin
    let xs = [1., 2., 3., 4., 5.],
        ys = [2.1, 3.9, 5.3, 7.7, 10.2],
        slope ~ Normal(0., 10.),
        intercept ~ Normal(0., 10.)
        [({:y => i} ~ Normal(intercept + slope * xs[i], 1.) ↦ ys[i]) for i in 1:5]
        (slope, intercept)
    end
end

const plated_linear_regression = @pgm plated graph_test_plated_linear_regression begin
    let xs = [1., 2., 3., 4., 5.],
        ys = [2.1, 3.9, 5.3, 7.7, 10.2],
        slope ~ Normal(0., 10.),
        intercept ~ Normal(0., 10.)
        [({:y => i} ~ Normal(intercept + slope * xs[i], 1.) ↦ ys[i]) for i in 1:5]
        (slope, intercept)
    end
end

# compiling runs the plate sampler, which crashed at random before C6 was fixed
const plated_gamma_poisson = @pgm plated graph_test_plated_gamma_poisson begin
    let ys = [2., 0., 5.],
        λ = [{:λ => i} ~ Gamma(i + 1., 1.) for i in 1:3]
        [({:y => i} ~ Poisson(λ[i]) ↦ ys[i]) for i in 1:3]
        λ
    end
end

# edges between unplated nodes and single plate members: a -> x[1] (via plate), x[1] -> z (C19)
const plated_single_member = @pgm plated graph_test_plated_single_member begin
    let a ~ Normal(0., 1.),
        xs = [{:x => i} ~ Normal(i == 1 ? a : 0., 1.) for i in 1:3],
        z ~ Normal(xs[1], 1.)
        {:y} ~ Normal(z, 1.) ↦ 0.5
        z
    end
end

# several latent plates and nodes: the plate order has to follow the program order,
# such that compiled likelihood weighting draws the same random numbers as likelihood weighting
const plated_mixture = @pgm plated graph_test_plated_mixture begin
    let ys = [-1., 0.5, 2.],
        σ = [{:σ => j} ~ Gamma(2., 1.) for j in 1:2],
        a ~ Normal(0., 1.),
        μ = [{:μ => j} ~ Normal(a, 1.) for j in 1:2],
        z = [{:z => i} ~ Categorical([0.5, 0.5]) for i in 1:3]
        [({:y => i} ~ Normal(μ[Int(z[i])], σ[Int(z[i])]) ↦ ys[i]) for i in 1:3]
        a
    end
end

# transformed (positive) variables read through a slice w[1:3] and a dynamic index s[Int(z)]
const unplated_transformed = @pgm uninvoked graph_test_unplated_transformed begin
    let w = [{:w => i} ~ Gamma(2., 1.) for i in 1:3],
        s = [{:s => i} ~ InverseGamma(2., 1.) for i in 1:3],
        z ~ Categorical(w / sum(w))
        {:y} ~ Normal(0., sqrt(s[Int(z)])) ↦ 0.5
        z
    end
end

# x only exists if c == 1
#   P(c = 1 | y) ∝ 0.3 N(y; 2, √2),   P(c = 0 | y) ∝ 0.7 N(y; -1, 1)
const lazy_branching = @pgm lazy_ifs graph_test_lazy_branching begin
    let c ~ Bernoulli(0.3),
        m = if c == 1.
                let x ~ Normal(2., 1.)
                    x
                end
            else
                -1.
            end
        {:y} ~ Normal(m, 1.) ↦ 0.8
        c
    end
end
const LAZY_POST_C1 = let l1 = 0.3 * exp(logpdf(Normal(2., sqrt(2.)), 0.8)), l0 = 0.7 * exp(logpdf(Normal(-1., 1.), 0.8))
    l1 / (l1 + l0)
end

# nested lazy branches: activating x (c = 1) may activate z (x > 0) (B17)
#   p(y | c = 1, x) = N(y; x, √2) if x > 0 (z integrated out), N(y; x, 1) otherwise
const lazy_nested = @pgm lazy_ifs graph_test_lazy_nested begin
    let c ~ Bernoulli(0.5),
        m = if c == 1.
                let x ~ Normal(0., 1.),
                    k = if x > 0.
                            let z ~ Normal(x, 1.)
                                z
                            end
                        else
                            x
                        end
                    k
                end
            else
                -1.
            end
        {:y} ~ Normal(m, 1.) ↦ 1.
        c
    end
end
const LAZY_NESTED_POST_C1, LAZY_NESTED_POST_Z_ACTIVE = let dx = 1e-4, xs = -10:dx:10
    w = pdf.(Normal(), xs) .* [x > 0 ? pdf(Normal(x, sqrt(2.)), 1.) : pdf(Normal(x, 1.), 1.) for x in xs]
    l1 = 0.5 * sum(w) * dx
    l0 = 0.5 * pdf(Normal(-1., 1.), 1.)
    l1 / (l1 + l0), 0.5 * sum(w[xs .> 0]) * dx / (l1 + l0)
end

# procedure call whose argument contains a sample statement
const procedure_sample_arg = @pgm graph_test_procedure_sample_arg begin
    function double(x)
        2 * x
    end
    let y = double({:x} ~ Normal(0., 1.))
        {:o} ~ Normal(y, 1.) ↦ 1.
        y
    end
end

node(pgm, addr) = findfirst(==(addr), pgm.addresses)
lr_means(traces) = [mean(traces[:slope]), mean(traces[:intercept])]

# -------------------------------------------------------------- deterministic

@testset "logjoint" begin
    s = 1.7
    ref = logpdf(Gamma(2., 1.), s) + sum(logpdf(Normal(0., s), y) for y in R.GS_YS)
    @test G.make_logjoint(gamma_scale)([s]) ≈ ref
    @test G.make_unconstrained_logjoint(gamma_scale)([log(s)]) ≈ ref + log(s) # C5

    s, b, x = 1.7, 0.4, -0.2
    ref = logpdf(Gamma(2., 1.), s) + logpdf(Uniform(0., s), b) + logpdf(Normal(b, s), x) + logpdf(Normal(x, 1.), 0.3)
    X = zeros(3)
    for (a, v) in ((:s, s), (:b, b), (:x, x))
        X[node(hierarchical_scale, a)] = v
    end
    @test G.make_logjoint(hierarchical_scale)(X) ≈ ref
    X_unconstrained = hierarchical_scale.transform_to_unconstrained!(X, similar(X))
    # log abs det of T^{-1}: s = exp(u) -> s, b = s * sigmoid(u) -> b (1 - b/s)
    @test G.make_unconstrained_logjoint(hierarchical_scale)(X_unconstrained) ≈ ref + log(s) + log(b * (1 - b / s)) # C5

    λ = [1.5, 0.7, 3.2]
    ref = sum(logpdf(Gamma(i + 1., 1.), λ[i]) + logpdf(Poisson(λ[i]), R.PGP_YS[i]) for i in 1:3)
    @test G.make_logjoint(plated_gamma_poisson)(λ) ≈ ref
    X = log.(λ)
    @test G.make_unconstrained_logjoint(plated_gamma_poisson)(X) ≈ ref + sum(log.(λ)) # C6
    @test X ≈ log.(λ) # B7: input must not be mutated
end

@testset "logjoint gradients with all AD backends" begin # B7
    function fd_grad(f, x; h=1e-6)
        return [(f(x .+ h .* (1:length(x) .== i)) - f(x .- h .* (1:length(x) .== i))) / 2h for i in eachindex(x)]
    end
    # plated / non-plated, constrained / unconstrained (plated unconstrained mutates a copy of X)
    for (pgm, values) in ((hierarchical_scale, (:s => 1.7, :b => 0.4, :x => -0.2)),
                          (plated_gamma_poisson, ((:λ => 1) => 1.5, (:λ => 2) => 0.7, (:λ => 3) => 3.2)),
                          (plated_linear_regression, (:slope => 0.8, :intercept => -0.3)))
        X = zeros(pgm.n_latents) # constrained values, lie in support
        for (addr, v) in values
            X[node(pgm, addr)] = v
        end
        for unconstrained in (false, true)
            if unconstrained
                logjoint = G.make_unconstrained_logjoint(pgm)
                X = pgm.transform_to_unconstrained!(X, similar(X))
            else
                logjoint = G.make_logjoint(pgm)
            end
            X_copy = copy(X)
            ref = fd_grad(logjoint, X)
            @testset "$(pgm.name) unconstrained=$unconstrained" begin
                @test Tracker.data(Tracker.gradient(logjoint, X)[1]) ≈ ref rtol = 1e-5
                @test ForwardDiff.gradient(logjoint, X) ≈ ref rtol = 1e-5
                @test ReverseDiff.gradient(logjoint, X) ≈ ref rtol = 1e-5
                @test X == X_copy # input is not mutated
            end
        end
    end
end

@testset "plated topological order with single member edges" begin # C19
    pgm = plated_single_member
    order = G.get_topological_order(pgm.n_variables, pgm.plate_info)
    position(node) = findfirst(u -> u isa G.Plate ? node in u.nodes : u == node, order)
    @test all(position(x) < position(y) for (x, y) in pgm.edges if position(x) != position(y))
    a, x, z = 0.3, [0.5, -0.2, 1.1], -0.4
    X = zeros(pgm.n_latents)
    X[node(pgm, :a)] = a
    X[node(pgm, :z)] = z
    for i in 1:3
        X[node(pgm, :x => i)] = x[i]
    end
    ref = logpdf(Normal(0., 1.), a) + logpdf(Normal(a, 1.), x[1]) + logpdf(Normal(0., 1.), x[2]) + logpdf(Normal(0., 1.), x[3]) +
        logpdf(Normal(x[1], 1.), z) + logpdf(Normal(z, 1.), 0.5)
    @test G.make_logjoint(pgm)(X) ≈ ref
end

@testset "dependencies" begin
    @test (node(lazy_branching, :c) => node(lazy_branching, :x)) in lazy_branching.edges # C8
end

@testset "procedure inlining keeps graph of arguments" begin # B6
    pgm = procedure_sample_arg
    @test pgm.addresses == Any[:x, :o]
    @test (node(pgm, :x) => node(pgm, :o)) in pgm.edges
    x = 0.3
    @test G.make_logjoint(pgm)([x]) ≈ logpdf(Normal(0., 1.), x) + logpdf(Normal(2x, 1.), 1.)
    @test G.get_retval(pgm, [x]) ≈ 2x
end

@testset "sample! respects support" begin # C6
    Random.seed!(0)
    X = Vector{Float64}(undef, plated_gamma_poisson.n_latents)
    samples = [(plated_gamma_poisson.sample!(X); copy(X)) for _ in 1:100]
    @test all(all(x .> 0) for x in samples)
end

# static parts of the program (ranges, observations, $(...)) are evaluated in the module where @pgm is called,
# and models can be created and used inside the same function (no world age issues) (J2, J4)
module GraphTestsOtherModule
    using TinyPPL.Graph, TinyPPL.Distributions
    ys = [1., 2.]
    n = 2
    shift = 10.
    function make_and_logpdf(x::Float64)
        scale = 1. # local variable, can be interpolated
        m = @pgm graph_test_other_module begin
            let x ~ Normal(0., 1.)
                [({:y => i} ~ Normal(x + $(shift), $(scale)) ↦ ys[i]) for i in 1:n]
                x
            end
        end
        lw = compile_likelihood_weighting(m)
        X = Vector{Float64}(undef, m.n_variables)
        lw(X, m.observations) # compiled in this function as well
        return m.observations, m.logpdf([x], m.observations)
    end
end

@testset "models in other modules and functions" begin # J2, J4
    observations, lp = GraphTestsOtherModule.make_and_logpdf(0.5)
    @test observations == [1., 2.]
    @test lp ≈ logpdf(Normal(0., 1.), 0.5) + logpdf(Normal(10.5, 1.), 1.) + logpdf(Normal(10.5, 1.), 2.)
end

@testset "model redefinition does not affect existing PGMs" begin # J2
    m1 = @pgm graph_test_redefine begin
        let x ~ Normal(0., 1.)
            x
        end
    end
    lp = m1.logpdf([0.5], Float64[])
    m2 = @pgm graph_test_redefine begin
        let x ~ Normal(100., 1.)
            x
        end
    end
    @test m1.logpdf([0.5], Float64[]) ≈ lp
    @test m1.distributions[1]([0.5]) == Normal(0., 1.)
    X = [0.]; m1.sample!(X)
    @test abs(X[1]) < 10 # from Normal(0., 1.), not Normal(100., 1.)

    # plated: the generated logpdf calls the plate functions by name
    p1 = @pgm plated graph_test_redefine_plated begin
        let xs = [{:x => i} ~ Normal(0., 1.) for i in 1:3]
            xs
        end
    end
    plp = p1.logpdf([0.5, 0.5, 0.5], Float64[])
    p2 = @pgm plated graph_test_redefine_plated begin
        let xs = [{:x => i} ~ Normal(100., 1.) for i in 1:3]
            xs
        end
    end
    @test p1.logpdf([0.5, 0.5, 0.5], Float64[]) ≈ plp ≈ 3 * logpdf(Normal(0., 1.), 0.5)
    @test p2.logpdf([0.5, 0.5, 0.5], Float64[]) ≈ 3 * logpdf(Normal(100., 1.), 0.5)
    lw_p1 = compile_likelihood_weighting(p1) # calls plate functions
    _, lp_p1 = lw_p1(zeros(3), Float64[])
    @test lp_p1 == 0. # no observations

    # compiled kernels of the same model
    lmh_kernels = compile_lmh(normal_normal)
    rwmh_kernels = compile_rwmh(normal_normal)
    @test all(lmh_kernels .!== rwmh_kernels)
    lw1 = compile_likelihood_weighting(m1)
    lw2 = compile_likelihood_weighting(m2)
    @test lw1 !== lw2
end

# ---------------------------------------------------------- likelihood weighting

@testset "likelihood_weighting" begin
    N = 50_000
    @testset "normal_normal" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(normal_normal, N)
        @test sum(exp, lw) ≈ 1 # normalised log weights (A1)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.03
        @test wstd(traces[:μ], lw) ≈ R.NN_POST_SD atol = 0.03
        @test wmean(G.retvals(traces), lw) ≈ R.NN_POST_MEAN atol = 0.03 # A8: retvals is not exported from Graph
        @test log_Z ≈ R.NN_LOG_EVIDENCE atol = 0.02
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(gamma_scale, N)
        @test wmean(traces[:s], lw) ≈ R.GS_POST_MEAN atol = 0.05
        @test log_Z ≈ R.GS_LOG_EVIDENCE atol = 0.02
    end
    @testset "beta_bernoulli" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(beta_bernoulli, N)
        @test wmean(traces[:p], lw) ≈ R.BB_POST_MEAN atol = 0.01
    end
    @testset "discrete_network" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(discrete_network, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.01
        @test wprob(traces[:b], lw, 1) ≈ R.DN_POST_B[2] atol = 0.01
        @test log_Z ≈ R.DN_LOG_EVIDENCE atol = 0.02
    end
    @testset "mixed" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(mixed, N)
        @test wprob(traces[:z], lw, 1) ≈ R.MX_POST_Z1 atol = 0.01
        @test wmean(traces[:x], lw) ≈ R.MX_POST_MEAN_X atol = 0.03
    end
    @testset "chain" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(chain, N)
        @test wmean(traces[:z], lw) ≈ R.CH_POST_MEAN_Z atol = 0.03
        @test log_Z ≈ R.CH_LOG_EVIDENCE atol = 0.02
    end
    @testset "plated_gamma_poisson" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(plated_gamma_poisson, N)
        @test [wmean(traces[:λ => i], lw) for i in 1:3] ≈ R.PGP_POST_MEAN atol = 0.1
    end
    @testset "lazy_branching" begin
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(lazy_branching, N)
        @test wprob(traces[:c], lw, 1) ≈ LAZY_POST_C1 atol = 0.01
    end
    @testset "compiled normal_normal" begin
        Random.seed!(0)
        lw_f = compile_likelihood_weighting(normal_normal)
        traces, lw, log_Z = compiled_likelihood_weighting(normal_normal, lw_f, N)
        @test sum(exp, lw) ≈ 1
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.03
        @test log_Z ≈ R.NN_LOG_EVIDENCE atol = 0.02
    end
    @testset "compiled equals generic (plated)" begin
        lw_f = compile_likelihood_weighting(plated_mixture)
        Random.seed!(0)
        traces, lw, log_Z = likelihood_weighting(plated_mixture, 1000)
        Random.seed!(0)
        compiled_traces, compiled_lw, compiled_log_Z = compiled_likelihood_weighting(plated_mixture, lw_f, 1000)
        @test traces.data == compiled_traces.data
        @test lw ≈ compiled_lw
        @test log_Z ≈ compiled_log_Z
    end
    @testset "compiled discrete_network" begin
        Random.seed!(0)
        lw_f = compile_likelihood_weighting(discrete_network)
        traces, lw, log_Z = compiled_likelihood_weighting(discrete_network, lw_f, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.01
    end
end

# ------------------------------------------------------------------ single-site MH

const MH_MODELS = [
    ("normal_normal", normal_normal, t -> begin
        @test mean(burn(t[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(t[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end),
    ("gamma_scale", gamma_scale, t -> @test mean(burn(t[:s])) ≈ R.GS_POST_MEAN atol = 0.1),
    ("beta_bernoulli", beta_bernoulli, t -> @test mean(burn(t[:p])) ≈ R.BB_POST_MEAN atol = 0.02),
    ("discrete_network", discrete_network, t -> begin
        @test frequencies(burn(t[:z]), 1:3) ≈ R.DN_POST_Z atol = 0.02
        @test mean(burn(t[:b])) ≈ R.DN_POST_B[2] atol = 0.02
    end),
    ("mixed", mixed, t -> begin
        @test mean(burn(t[:z])) ≈ R.MX_POST_Z1 atol = 0.03
        @test mean(burn(t[:x])) ≈ R.MX_POST_MEAN_X atol = 0.05
    end),
    ("chain", chain, t -> @test mean(burn(t[:z])) ≈ R.CH_POST_MEAN_Z atol = 0.05),
    ("plated_gamma_poisson", plated_gamma_poisson, t -> @test [mean(burn(t[:λ => i])) for i in 1:3] ≈ R.PGP_POST_MEAN atol = 0.15),
    ("lazy_branching", lazy_branching, t -> begin # B17, C8
        c, x = burn(t[:c]), burn(t[:x])
        @test mean(c) ≈ LAZY_POST_C1 atol = 0.03
        @test all(isnan.(x) .== (c .== 0)) # x is inactive iff c == 0
        @test mean(x[c .== 1]) ≈ (2. + 0.8) / 2 atol = 0.05 # x | c = 1, y ~ N(1.4, 1/2)
    end),
    ("lazy_nested", lazy_nested, t -> begin # B17
        c, x, z = burn(t[:c]), burn(t[:x]), burn(t[:z])
        @test mean(c) ≈ LAZY_NESTED_POST_C1 atol = 0.03
        @test mean(.!isnan.(z)) ≈ LAZY_NESTED_POST_Z_ACTIVE atol = 0.03
        @test all(isnan.(z) .== .!((c .== 1) .& (x .> 0))) # z is active iff c == 1 and x > 0
    end),
]

@testset "trace indexing with ranges and masks" begin
    Random.seed!(0)
    traces, _, _ = likelihood_weighting(normal_normal, 100)
    test_trace_indexing(traces, :μ)
end

@testset "lmh" begin
    for (name, pgm, check) in MH_MODELS
        @testset "$name" begin
            Random.seed!(0)
            check(lmh(pgm, 100_000))
        end
    end
end

@testset "rwmh" begin
    for (name, pgm, check) in MH_MODELS
        @testset "$name" begin
            Random.seed!(0)
            check(rwmh(pgm, 100_000; default_var=0.5))
        end
    end
end

@testset "compiled single site" begin
    @testset "compile_lmh normal_normal" begin
        Random.seed!(0)
        traces = compiled_single_site(normal_normal, compile_lmh(normal_normal), 30_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
    end
    @testset "compile_lmh discrete_network" begin
        Random.seed!(0)
        traces = compiled_single_site(discrete_network, compile_lmh(discrete_network), 60_000)
        @test frequencies(burn(traces[:z]), 1:3) ≈ R.DN_POST_Z atol = 0.02
    end
    @testset "compile_rwmh mixed" begin
        Random.seed!(0)
        traces = compiled_single_site(mixed, compile_rwmh(mixed; default_var=0.5), 60_000)
        @test mean(burn(traces[:z])) ≈ R.MX_POST_Z1 atol = 0.03
    end
    @testset "lazy_ifs models are rejected" begin # B17
        @test_throws ErrorException compile_lmh(lazy_branching)
        @test_throws ErrorException compile_rwmh(lazy_branching)
        @test_throws ErrorException G.single_site(lazy_branching, 10, (X, node) -> 0.)
        @test !G.has_lazy_latents(normal_normal)
        @test G.has_lazy_latents(lazy_branching)
    end
    @testset "compile_lmh with proposal" begin # B5
        Random.seed!(0)
        kernels = compile_lmh(normal_normal; addr2proposal=Addr2Proposal(:μ => ContinuousRandomWalkProposal(0.5)))
        traces = compiled_single_site(normal_normal, kernels, 30_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
    end
end

# ---------------------------------------------------------------------- HMC

@testset "hmc" begin
    @testset "normal_normal" begin
        Random.seed!(0)
        traces = hmc(normal_normal, 3_000, 10, 0.1)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(traces[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "linear_regression" begin
        Random.seed!(0)
        traces = hmc(linear_regression, 3_000, 10, 0.05)
        @test lr_means(traces[:, 301:3000]) ≈ R.LR_POST_MEAN atol = 0.1
    end
    @testset "plated_linear_regression" begin # B7
        Random.seed!(0)
        traces = hmc(plated_linear_regression, 3_000, 10, 0.05)
        @test lr_means(traces[:, 301:3000]) ≈ R.LR_POST_MEAN atol = 0.1
    end
    @testset "gamma_scale unconstrained" begin # C5
        Random.seed!(0)
        x_initial = [1.0]
        traces = hmc(gamma_scale, 3_000, 10, 0.1; unconstrained=true, x_initial=x_initial)
        @test x_initial == [1.0] # A6
        @test all(traces[:s] .> 0)
        @test mean(burn(traces[:s])) ≈ R.GS_POST_MEAN atol = 0.1
    end
    @testset "plated_gamma_poisson unconstrained" begin # B7, C6
        Random.seed!(0)
        traces = hmc(plated_gamma_poisson, 3_000, 10, 0.1; unconstrained=true)
        @test [mean(burn(traces[:λ => i])) for i in 1:3] ≈ R.PGP_POST_MEAN atol = 0.15
    end
    @testset "forwarddiff backend" begin # B7
        Random.seed!(0)
        traces = hmc(normal_normal, 1_000, 10, 0.1; ad_backend=:forwarddiff)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.1
    end
end

# ---------------------------------------------------------------- variational

@testset "advi" begin
    @testset "advi_meanfield normal_normal" begin
        Random.seed!(0)
        traces = sample_posterior(advi_meanfield(normal_normal, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(traces[:μ]) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "advi_meanfield gamma_scale" begin # C5
        Random.seed!(0)
        traces = sample_posterior(advi_meanfield(gamma_scale, 10_000, 10, 0.01), 10_000)
        @test all(traces[:s] .> 0)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
    end
    @testset "advi_meanfield chain" begin
        Random.seed!(0)
        traces = sample_posterior(advi_meanfield(chain, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:z]) ≈ R.CH_POST_MEAN_Z atol = 0.1
    end
    @testset "advi_fullrank linear_regression" begin
        Random.seed!(0)
        traces = sample_posterior(advi_fullrank(linear_regression, 10_000, 10, 0.01), 10_000)
        @test lr_means(traces) ≈ R.LR_POST_MEAN atol = 0.1
        @test [std(traces[:slope]), std(traces[:intercept])] ≈ R.LR_POST_SD atol = 0.1
    end
    @testset "advi with estimator normal_normal" begin
        Random.seed!(0)
        res = advi(normal_normal, 10_000, 10, 0.01, MeanFieldGaussian(1), RelativeEntropyELBO())
        @test mean(sample_posterior(res, 10_000)[:μ]) ≈ R.NN_POST_MEAN atol = 0.05
    end
    @testset "advi with estimator gamma_scale unconstrained" begin # B5, C5
        Random.seed!(0)
        res = advi(gamma_scale, 10_000, 10, 0.01, MeanFieldGaussian(1), RelativeEntropyELBO(); unconstrained=true)
        traces = sample_posterior(res, 10_000)
        @test all(traces[:s] .> 0)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
    end
end

@testset "bbvi" begin
    @testset "bbvi normal_normal" begin # B8
        Random.seed!(0)
        traces = sample_posterior(bbvi(normal_normal, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.1
    end
    @testset "bbvi_naive chain" begin # B8, C7
        Random.seed!(0)
        traces = sample_posterior(bbvi_naive(chain, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:z]) ≈ R.CH_POST_MEAN_Z atol = 0.1
    end
    @testset "bbvi_rao chain" begin # B8, C7
        Random.seed!(0)
        traces = sample_posterior(bbvi_rao(chain, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:z]) ≈ R.CH_POST_MEAN_Z atol = 0.1
    end
end

# ---------------------------------------------------------------------- SMC

@testset "smc" begin
    N = 20_000
    @testset "smc normal_normal" begin
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(normal_normal, N)
        @test sum(exp, lw) ≈ 1 # normalised log weights (A1)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.05
        @test log_marginal_lik ≈ R.NN_LOG_EVIDENCE atol = 0.05
    end
    @testset "smc discrete_network" begin
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(discrete_network, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.02
        @test log_marginal_lik ≈ R.DN_LOG_EVIDENCE atol = 0.05
    end
    @testset "smc mixed" begin
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(mixed, N)
        @test wprob(traces[:z], lw, 1) ≈ R.MX_POST_Z1 atol = 0.02
        @test log_marginal_lik ≈ R.MX_LOG_EVIDENCE atol = 0.05
    end
    @testset "smc with proposal" begin # C14
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(normal_normal, N; addr2proposal=Addr2Proposal(:μ => StaticProposal(Normal(1., 1.))))
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.05
        @test log_marginal_lik ≈ R.NN_LOG_EVIDENCE atol = 0.05
    end
    @testset "smc with proposal for latent after last observe" begin # C14
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(normal_normal_trailing, N; addr2proposal=Addr2Proposal(:w => StaticProposal(Normal(2., 2.))))
        @test wmean(traces[:w], lw) ≈ R.NN_POST_MEAN atol = 0.05
        @test wstd(traces[:w], lw) ≈ sqrt(R.NN_POST_SD^2 + 1) atol = 0.05
        @test log_marginal_lik ≈ R.NN_LOG_EVIDENCE atol = 0.05
    end
    @testset "light_smc chain" begin # C12
        Random.seed!(0)
        traces, lw, log_marginal_lik = light_smc(chain, N)
        @test all(lw .≈ -log(N)) # particles are resampled at last observation
        @test mean(traces[:z]) ≈ R.CH_POST_MEAN_Z atol = 0.05
        @test log_marginal_lik ≈ R.CH_LOG_EVIDENCE atol = 0.05
    end
end

@testset "conditional_smc" begin # T2
    pgm = normal_normal_trailing
    X_ref = zeros(pgm.n_latents)
    X_ref[node(pgm, :μ)], X_ref[node(pgm, :w)] = 0.3, -0.7
    @testset "reference particle is retained" begin
        Random.seed!(0)
        traces, lw, log_Z = conditional_smc(pgm, 10, X_ref)
        @test traces[:, 1] == X_ref
        @test traces.retvals[1] == -0.7
        @test sum(exp, lw) ≈ 1 && isfinite(log_Z)
    end
    # conditional SMC leaves the posterior invariant: if X_ref ~ p(X|Y), then a particle picked by weight ~ p(X|Y)
    @testset "posterior invariance ancestral_sampling=$ancestral_sampling" for ancestral_sampling in (false, true)
        Random.seed!(0)
        n = 5_000
        μs, ws = zeros(n), zeros(n)
        for j in 1:n
            μ = rand(Normal(R.NN_POST_MEAN, R.NN_POST_SD))
            X_ref[node(pgm, :μ)], X_ref[node(pgm, :w)] = μ, rand(Normal(μ, 1.))
            traces, lw, _ = conditional_smc(pgm, 5, X_ref; ancestral_sampling=ancestral_sampling)
            k = rand(Categorical(exp.(lw)))
            μs[j], ws[j] = traces[:μ, k], traces[:w, k]
        end
        @test mean(μs) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(μs) ≈ R.NN_POST_SD atol = 0.05
        @test mean(ws) ≈ R.NN_POST_MEAN atol = 0.08
        @test std(ws) ≈ sqrt(R.NN_POST_SD^2 + 1) atol = 0.08
    end
end

@testset "particle MCMC" begin
    @testset "particle_gibbs normal_normal" begin
        Random.seed!(0)
        traces = particle_gibbs(normal_normal, 10, 2_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(traces[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "particle_gibbs discrete_network" begin
        Random.seed!(0)
        traces = particle_gibbs(discrete_network, 10, 2_000)
        @test frequencies(burn(traces[:z]), 1:3) ≈ R.DN_POST_Z atol = 0.05
    end
    @testset "pgas chain" begin # C13
        Random.seed!(0)
        traces = particle_gibbs(chain, 10, 2_000; ancestral_sampling=true)
        @test mean(burn(traces[:z])) ≈ R.CH_POST_MEAN_Z atol = 0.1
    end
    @testset "pgas future_parents" begin # C13
        Random.seed!(0)
        traces = particle_gibbs(future_parents, 10, 5_000; ancestral_sampling=true)
        @test mean(burn(traces[:z])) ≈ FP_POST_MEAN[1] atol = 0.05
        @test mean(burn(traces[:a])) ≈ FP_POST_MEAN[2] atol = 0.05
    end
    @testset "pgas linear_regression" begin # C13
        Random.seed!(0)
        # with 10 particles the chain mixes slowly (sd of the intercept estimate ≈ 0.18 at 2000 iterations),
        # with 50 particles and 4000 iterations the sd across seeds is ≈ 0.02 (slope) and ≈ 0.08 (intercept)
        traces = particle_gibbs(linear_regression, 50, 4_000; ancestral_sampling=true)
        @test lr_means(traces[:, 401:4000]) ≈ R.LR_POST_MEAN atol = 0.25
    end
    @testset "particle_IMH normal_normal" begin # B5
        Random.seed!(0)
        traces = particle_IMH(normal_normal, 10, 2_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
    end
end


@testset "traces API" begin
    # A7: integer index is an address, not a node index
    traces = G.GraphTraces(Dict{Any,Int}(2 => 1, :x => 2), [10. 11.; 20. 21.], Any[nothing, nothing])
    @test traces[2] == [10., 11.]
    @test traces[:x, 2] == 21.
    @test_throws AssertionError traces[1]
    # A8: one retvals / subset function for all backends
    @test G.retvals === TinyPPL.Evaluation.retvals === TinyPPL.retvals
    @test G.subset === TinyPPL.Evaluation.subset
    @test length(subset(traces, [2])) == 1
end

@testset "unconstrained logpdf of transformed variables" begin
    # Gamma and InverseGamma are transformed with log, so log p(y) = log p(x) + Σ y_i over the transformed nodes.
    # Before, children read the unconstrained values through slices and dynamic indices (log w can be negative).
    m = unplated_transformed
    transformed = [i for i in 1:m.n_latents if m.addresses[i] isa Pair]
    Random.seed!(0)
    X = zeros(m.n_latents); Y = zeros(m.n_latents)
    all_equal = true
    for _ in 1:500
        m.sample!(X)
        m.transform_to_unconstrained!(X, Y)
        all_equal &= m.unconstrained_logpdf(Y, m.observations) ≈ m.logpdf(X, m.observations) + sum(Y[transformed])
    end
    @test all_equal
end

end # module GraphTests
