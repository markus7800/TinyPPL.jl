module GraphTests
# Graph-based (PGM) approximate inference (src/graph, without src/graph/exact).

using Test, Random, Statistics
using TinyPPL.Distributions
using TinyPPL.Graph
import TinyPPL.Graph as G
import ..References as R
using ..TestUtils

# ---------------------------------------------------------------------- models
# see references.jl for the mathematical definitions.
# Data is inlined, since @pgm cannot see variables of this module.
# Model names must be globally unique (REVIEW.md J2), hence the `graph_test_` prefix.

const normal_normal = @pgm graph_test_normal_normal begin
    let μ ~ Normal(0., 2.),
        ys = [1.0, 2.0, 1.5]
        [({:y => i} ~ Normal(μ, 1.) ↦ ys[i]) for i in 1:3]
        μ
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

@testset "dependencies" begin
    @test (node(lazy_branching, :c) => node(lazy_branching, :x)) in lazy_branching.edges # C8
end

@testset "sample! respects support" begin # C6
    Random.seed!(0)
    X = Vector{Float64}(undef, plated_gamma_poisson.n_latents)
    samples = [(plated_gamma_poisson.sample!(X); copy(X)) for _ in 1:100]
    @test all(all(x .> 0) for x in samples)
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
end

# ---------------------------------------------------------- likelihood weighting

@testset "likelihood_weighting" begin
    N = 50_000
    @testset "normal_normal" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(normal_normal, N)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.03
        @test wstd(traces[:μ], lw) ≈ R.NN_POST_SD atol = 0.03
        @test wmean(G.retvals(traces), lw) ≈ R.NN_POST_MEAN atol = 0.03 # A8: retvals is not exported from Graph
        @test is_log_evidence(lw) ≈ R.NN_LOG_EVIDENCE atol = 0.02
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(gamma_scale, N)
        @test wmean(traces[:s], lw) ≈ R.GS_POST_MEAN atol = 0.05
        @test is_log_evidence(lw) ≈ R.GS_LOG_EVIDENCE atol = 0.02
    end
    @testset "beta_bernoulli" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(beta_bernoulli, N)
        @test wmean(traces[:p], lw) ≈ R.BB_POST_MEAN atol = 0.01
    end
    @testset "discrete_network" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(discrete_network, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.01
        @test wprob(traces[:b], lw, 1) ≈ R.DN_POST_B[2] atol = 0.01
        @test is_log_evidence(lw) ≈ R.DN_LOG_EVIDENCE atol = 0.02
    end
    @testset "mixed" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(mixed, N)
        @test wprob(traces[:z], lw, 1) ≈ R.MX_POST_Z1 atol = 0.01
        @test wmean(traces[:x], lw) ≈ R.MX_POST_MEAN_X atol = 0.03
    end
    @testset "chain" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(chain, N)
        @test wmean(traces[:z], lw) ≈ R.CH_POST_MEAN_Z atol = 0.03
        @test is_log_evidence(lw) ≈ R.CH_LOG_EVIDENCE atol = 0.02
    end
    @testset "plated_gamma_poisson" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(plated_gamma_poisson, N)
        @test [wmean(traces[:λ => i], lw) for i in 1:3] ≈ R.PGP_POST_MEAN atol = 0.1
    end
    @testset "lazy_branching" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(lazy_branching, N)
        @test wprob(traces[:c], lw, 1) ≈ LAZY_POST_C1 atol = 0.01
    end
    @testset "compiled normal_normal" begin
        Random.seed!(0)
        lw_f = compile_likelihood_weighting(normal_normal)
        traces, lw = compiled_likelihood_weighting(normal_normal, lw_f, N)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.03
        @test is_log_evidence(lw) ≈ R.NN_LOG_EVIDENCE atol = 0.02
    end
    @testset "compiled discrete_network" begin
        Random.seed!(0)
        lw_f = compile_likelihood_weighting(discrete_network)
        traces, lw = compiled_likelihood_weighting(discrete_network, lw_f, N)
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
    ("lazy_branching", lazy_branching, t -> @test mean(burn(t[:c])) ≈ LAZY_POST_C1 atol = 0.03), # B16, C8
]

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
    @testset "compile_lmh with proposal" begin # B5
        Random.seed!(0)
        kernels = compile_lmh(normal_normal; addr2Proposal=Addr2Proposal(:μ => ContinuousRandomWalkProposal(0.5)))
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
    @testset "light_smc chain" begin # C12
        Random.seed!(0)
        traces, lw, marginal_lik = light_smc(chain, N)
        @test wmean(traces[:z], lw) ≈ R.CH_POST_MEAN_Z atol = 0.05
        @test log(marginal_lik) ≈ R.CH_LOG_EVIDENCE atol = 0.05
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
    @testset "pgas linear_regression" begin # C13
        Random.seed!(0)
        traces = particle_gibbs(linear_regression, 10, 2_000; ancestral_sampling=true)
        @test lr_means(traces[:, 201:2000]) ≈ R.LR_POST_MEAN atol = 0.2
    end
    @testset "particle_IMH normal_normal" begin # B5
        Random.seed!(0)
        traces = particle_IMH(normal_normal, 10, 2_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
    end
end

end # module GraphTests
