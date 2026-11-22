module StaticEvaluationTests
# Evaluation-based inference for static models (src/evaluation/static).

using Test, Random, Statistics
using TinyPPL.Distributions
using TinyPPL.Evaluation
import TinyPPL.Evaluation as E
import Tracker, ForwardDiff, ReverseDiff
import ..References as R
using ..TestUtils

# ---------------------------------------------------------------------- models
# see references.jl for the mathematical definitions

@ppl static function normal_normal()
    μ ~ Normal(0., R.NN_PRIOR_SD)
    for i in 1:3
        {:y => i} ~ Normal(μ, 1.)
    end
    return μ
end
const nn_obs = Observations((:y => i) => R.NN_YS[i] for i in 1:3)

# normal_normal with a latent variable after the last observation
@ppl static function normal_normal_trailing()
    μ ~ Normal(0., R.NN_PRIOR_SD)
    for i in 1:3
        {:y => i} ~ Normal(μ, 1.)
    end
    z ~ Normal(μ, 1.)
    return z
end

@ppl static function gamma_scale()
    s ~ Gamma(2., 1.)
    for i in 1:3
        {:y => i} ~ Normal(0., s)
    end
    return s
end
const gs_obs = Observations((:y => i) => R.GS_YS[i] for i in 1:3)

@ppl static function beta_bernoulli()
    p ~ Beta(2., 2.)
    for i in 1:5
        {:y => i} ~ Bernoulli(p)
    end
    return p
end
const bb_obs = Observations((:y => i) => R.BB_YS[i] for i in 1:5)

@ppl static function discrete_network()
    z ~ Categorical([0.5, 0.3, 0.2])
    b ~ Bernoulli(R.dn_pb(z))
    {:o1} ~ Bernoulli(R.dn_po1(b))
    {:o2} ~ Bernoulli(R.dn_po2(z))
    return z
end
const dn_obs = Observations(:o1 => 1., :o2 => 0.)

@ppl static function mixed()
    z ~ Bernoulli(0.3)
    x ~ Normal(R.mx_mean(z), 1.)
    {:y} ~ Normal(x, 0.5)
    return x
end
const mx_obs = Observations(:y => R.MX_Y)

@ppl static function linear_regression(xs)
    slope ~ Normal(0., 10.)
    intercept ~ Normal(0., 10.)
    for i in eachindex(xs)
        {:y => i} ~ Normal(intercept + slope * xs[i], 1.)
    end
    return (slope, intercept)
end
const lr_obs = Observations((:y => i) => R.LR_YS[i] for i in eachindex(R.LR_YS))
const lr_args = (R.LR_XS,)

lr_means(traces) = [mean(traces[:slope]), mean(traces[:intercept])]

# -------------------------------------------------------------- deterministic

@testset "logjoint" begin
    lj, addresses_to_ix = E.make_logjoint(gamma_scale, (), gs_obs)
    @test addresses_to_ix == Dict(:s => 1)
    s = 1.7
    ref = logpdf(Gamma(2., 1.), s) + sum(logpdf(Normal(0., s), y) for y in R.GS_YS)
    @test lj([s]) ≈ ref

    ulj, _ = E.make_unconstrained_logjoint(gamma_scale, (), gs_obs)
    @test ulj([log(s)]) ≈ ref + log(s)

    X = [s]
    E.transform_to_unconstrained!(X, gamma_scale, (), gs_obs, addresses_to_ix)
    @test X ≈ [log(s)]
    _, retval = E.transform_to_constrained!(X, gamma_scale, (), gs_obs, addresses_to_ix)
    @test X ≈ [s]
    @test retval ≈ s
end

# ---------------------------------------------------------- likelihood weighting

@testset "likelihood_weighting" begin
    N = 50_000
    @testset "normal_normal" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(normal_normal, (), nn_obs, N)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.03
        @test wstd(traces[:μ], lw) ≈ R.NN_POST_SD atol = 0.03
        @test is_log_evidence(lw) ≈ R.NN_LOG_EVIDENCE atol = 0.02
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(gamma_scale, (), gs_obs, N)
        @test wmean(traces[:s], lw) ≈ R.GS_POST_MEAN atol = 0.05
        @test is_log_evidence(lw) ≈ R.GS_LOG_EVIDENCE atol = 0.02
    end
    @testset "beta_bernoulli" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(beta_bernoulli, (), bb_obs, N)
        @test wmean(traces[:p], lw) ≈ R.BB_POST_MEAN atol = 0.01
        @test is_log_evidence(lw) ≈ R.BB_LOG_EVIDENCE atol = 0.02
    end
    @testset "discrete_network" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(discrete_network, (), dn_obs, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.01
        @test is_log_evidence(lw) ≈ R.DN_LOG_EVIDENCE atol = 0.02
    end
    @testset "mixed" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(mixed, (), mx_obs, N)
        @test wprob(traces[:z], lw, 1) ≈ R.MX_POST_Z1 atol = 0.01
        @test wmean(traces[:x], lw) ≈ R.MX_POST_MEAN_X atol = 0.03
        @test is_log_evidence(lw) ≈ R.MX_LOG_EVIDENCE atol = 0.02
    end
    @testset "linear_regression with address subset" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(linear_regression, lr_args, lr_obs, 200_000, Address[:slope])
        @test size(traces.data, 1) == 1
        @test wmean(traces[:slope], lw) ≈ R.LR_POST_MEAN[1] atol = 0.1
        @test is_log_evidence(lw) ≈ R.LR_LOG_EVIDENCE atol = 0.2
    end
end

# ---------------------------------------------------------------------- HMC

@testset "hmc" begin
    @testset "normal_normal" begin
        Random.seed!(0)
        traces = hmc(normal_normal, (), nn_obs, 3_000, 10, 0.1)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(traces[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "gamma_scale constrained" begin # C4
        Random.seed!(0)
        traces = hmc(gamma_scale, (), gs_obs, 3_000, 10, 0.05; x_initial=[1.0])
        @test all(traces[:s] .> 0)
        @test mean(burn(traces[:s])) ≈ R.GS_POST_MEAN atol = 0.1
        @test retvals(traces) == traces[:s]
    end
    @testset "gamma_scale constrained default initialisation" begin # C4
        Random.seed!(0)
        traces = hmc(gamma_scale, (), gs_obs, 3_000, 10, 0.05)
        @test mean(burn(traces[:s])) ≈ R.GS_POST_MEAN atol = 0.1
    end
    @testset "gamma_scale unconstrained" begin
        Random.seed!(0)
        x_initial = [1.0]
        traces = hmc(gamma_scale, (), gs_obs, 3_000, 10, 0.1; unconstrained=true, x_initial=x_initial)
        @test x_initial == [1.0] # A6
        @test mean(burn(traces[:s])) ≈ R.GS_POST_MEAN atol = 0.1
    end
    @testset "linear_regression" begin
        Random.seed!(0)
        traces = hmc(linear_regression, lr_args, lr_obs, 3_000, 10, 0.05)
        @test lr_means(subset(traces, 301:3000)) ≈ R.LR_POST_MEAN atol = 0.1
    end
end

# ---------------------------------------------------------------- variational

@testset "advi_meanfield" begin
    @testset "normal_normal" begin
        Random.seed!(0)
        traces = sample_posterior(advi_meanfield(normal_normal, (), nn_obs, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(traces[:μ]) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        traces = sample_posterior(advi_meanfield(gamma_scale, (), gs_obs, 10_000, 10, 0.01), 10_000)
        @test all(traces[:s] .> 0)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
    end
    @testset "linear_regression" begin
        Random.seed!(0)
        traces = sample_posterior(advi_meanfield(linear_regression, lr_args, lr_obs, 10_000, 10, 0.01), 10_000)
        @test lr_means(traces) ≈ R.LR_POST_MEAN atol = 0.1
    end
end

@testset "advi_fullrank" begin
    @testset "linear_regression" begin
        Random.seed!(0)
        traces = sample_posterior(advi_fullrank(linear_regression, lr_args, lr_obs, 10_000, 10, 0.01), 10_000)
        @test lr_means(traces) ≈ R.LR_POST_MEAN atol = 0.1
        @test [std(traces[:slope]), std(traces[:intercept])] ≈ R.LR_POST_SD atol = 0.1
    end
end

# guides (B14)
@ppl static function normal_normal_guide()
    m = param("m")
    s = param("s", constraint=Positive())
    μ ~ Normal(m, s)
end
# guide that stores a parameter directly as value (ReverseDiff: element of tracked array with origin)
@ppl static function dirac_guide()
    m = param("m")
    s = param("s", constraint=Positive())
    a ~ Dirac(m)
    b ~ Normal(m, s)
end
# guide for unconstrained gamma_scale (values of s are unconstrained)
@ppl static function gamma_scale_guide()
    m = param("m")
    s = param("s", constraint=Positive())
    s ~ Normal(m, s)
end

@testset "guides" begin # B14
    q = E.make_guide(normal_normal_guide, (), Dict{Any,Int}(:μ => 1))
    phi = [0.3, -0.2]
    function f(p)
        Random.seed!(1)
        x, lp = E.rand_and_logpdf(E.update_params(q, p))
        return x[1]^2 + lp
    end
    @testset "rand_and_logpdf gradients agree for all AD backends" begin
        g = Tracker.data(Tracker.gradient(f, phi)[1])
        @test ForwardDiff.gradient(f, phi) ≈ g
        @test ReverseDiff.gradient(f, phi) ≈ g
        # reparametrised sample μ = m + exp(ω) ε, so d/dm (μ^2 + logpdf) = 2μ
        Random.seed!(1); μ = E.rand(E.update_params(q, phi))[1]
        @test g[1] ≈ 2μ
    end
    @testset "concrete value types" begin
        rcfg = ReverseDiff.GradientConfig(phi); ReverseDiff.track!(rcfg.input, phi)
        for p in (phi, Tracker.param(phi), ForwardDiff.GradientConfig(nothing, phi).duals, rcfg.input)
            sampler = E.update_params(q, p).sampler
            @test isconcretetype(eltype(sampler.X)) && isconcretetype(typeof(sampler.W))
        end
    end
    @testset "parameter stored as value" begin
        q_dirac = E.make_guide(dirac_guide, (), Dict{Any,Int}(:a => 1, :b => 2))
        function f_dirac(p)
            Random.seed!(1)
            x = rand(E.update_params(q_dirac, p))
            return 3 * x[1] + x[2]^2
        end
        g = Tracker.data(Tracker.gradient(f_dirac, phi)[1])
        @test ForwardDiff.gradient(f_dirac, phi) ≈ g
        @test ReverseDiff.gradient(f_dirac, phi) ≈ g
        Random.seed!(1); b = rand(E.update_params(q_dirac, phi))[2]
        @test g[1] ≈ 3 + 2b # a = m, b = m + exp(ω) ε
    end
    @testset "logpdf" begin
        q_phi = E.update_params(q, phi)
        X = [0.7]
        @test logpdf(q_phi, X) ≈ logpdf(Normal(0.3, exp(-0.2)), 0.7)
        @test logpdf(q_phi, view([0.1, 0.7], 2:2)) ≈ logpdf(Normal(0.3, exp(-0.2)), 0.7)
        @test eltype(rand(q_phi, 5)) == Float64
    end
    for backend in (:tracker, :forwarddiff, :reversediff)
        @testset "advi normal_normal $backend" begin
            Random.seed!(0)
            res = advi(normal_normal, (), nn_obs, 2_000, 10, 0.05, normal_normal_guide, (), MonteCarloELBO(); ad_backend=backend)
            p = E.get_constrained_parameters(res.Q)
            @test p["m"] ≈ R.NN_POST_MEAN atol = 0.05
            @test p["s"] ≈ R.NN_POST_SD atol = 0.05
            traces = sample_posterior(res, 10_000)
            @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.05
        end
    end
    @testset "advi gamma_scale unconstrained" begin
        Random.seed!(0)
        res = advi(gamma_scale, (), gs_obs, 5_000, 10, 0.05, gamma_scale_guide, (), MonteCarloELBO(); unconstrained=true)
        traces = sample_posterior(res, 10_000)
        @test all(traces[:s] .> 0)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
    end
end

@testset "advi with estimator" begin
    for estimator in (MonteCarloELBO(), RelativeEntropyELBO(), PathDerivativeELBO())
        @testset "normal_normal $(typeof(estimator))" begin
            Random.seed!(0)
            res = advi(normal_normal, (), nn_obs, 10_000, 10, 0.01, MeanFieldGaussian(1), estimator)
            traces = sample_posterior(res, 10_000)
            @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.05
        end
        @testset "gamma_scale unconstrained $(typeof(estimator))" begin # B5
            Random.seed!(0)
            res = advi(gamma_scale, (), gs_obs, 10_000, 10, 0.01, MeanFieldGaussian(1), estimator; unconstrained=true)
            traces = sample_posterior(res, 10_000)
            @test all(traces[:s] .> 0)
            @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
        end
    end
    @testset "forwarddiff backend" begin
        Random.seed!(0)
        res = advi(linear_regression, lr_args, lr_obs, 10_000, 10, 0.01, MeanFieldGaussian(2), RelativeEntropyELBO(); ad_backend=:forwarddiff)
        @test lr_means(sample_posterior(res, 10_000)) ≈ R.LR_POST_MEAN atol = 0.1
    end
end

@testset "bbvi" begin
    @testset "normal_normal" begin
        Random.seed!(0)
        traces = sample_posterior(bbvi(normal_normal, (), nn_obs, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.1
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        traces = sample_posterior(bbvi(gamma_scale, (), gs_obs, 10_000, 10, 0.01), 10_000)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
    end
end

# ---------------------------------------------------------------------- SMC

@testset "smc" begin
    N = 5_000
    @testset "normal_normal" begin # C3
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(normal_normal, (), nn_obs, N)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.05
        @test log_marginal_lik ≈ R.NN_LOG_EVIDENCE atol = 0.05
    end
    @testset "discrete_network" begin # C3
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(discrete_network, (), dn_obs, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.02
        @test log_marginal_lik ≈ R.DN_LOG_EVIDENCE atol = 0.05
    end
    @testset "mixed" begin # C3
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(mixed, (), mx_obs, N)
        @test wprob(traces[:z], lw, 1) ≈ R.MX_POST_Z1 atol = 0.02
        @test log_marginal_lik ≈ R.MX_LOG_EVIDENCE atol = 0.05
    end
    @testset "latent after last observation with proposal" begin # C3
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(normal_normal_trailing, (), nn_obs, N;
            addr2proposal=Addr2Proposal(:z => StaticProposal(Normal(2., 2.))))
        @test log_marginal_lik ≈ R.NN_LOG_EVIDENCE atol = 0.05
        @test wmean(traces[:z], lw) ≈ R.NN_POST_MEAN atol = 0.05
        @test wstd(traces[:z], lw) ≈ sqrt(R.NN_POST_SD^2 + 1) atol = 0.05
    end
    @testset "evidence of many observations does not underflow" begin # C3
        Random.seed!(0)
        traces, lw, log_marginal_lik = smc(normal_normal, (), Observations((:y => i) => 100. for i in 1:3), 1_000)
        @test isfinite(log_marginal_lik)
    end
end

@testset "particle MCMC" begin
    @testset "particle_gibbs normal_normal" begin
        Random.seed!(0)
        traces = particle_gibbs(normal_normal, (), nn_obs, 10, 1_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(traces[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "particle_gibbs discrete_network" begin
        Random.seed!(0)
        traces = particle_gibbs(discrete_network, (), dn_obs, 10, 1_000)
        @test frequencies(burn(traces[:z]), 1:3) ≈ R.DN_POST_Z atol = 0.05
    end
    @testset "pgas linear_regression" begin
        Random.seed!(0)
        traces = particle_gibbs(linear_regression, lr_args, lr_obs, 10, 1_000; ancestral_sampling=true)
        # chain mixes slowly (posterior sd of intercept ≈ 1), with 20k samples PG/PGAS/PIMH are within 0.1
        slope, intercept = lr_means(subset(traces, 101:1000))
        @test slope ≈ R.LR_POST_MEAN[1] atol = 0.15
        @test intercept ≈ R.LR_POST_MEAN[2] atol = 0.4
    end
    @testset "particle_IMH normal_normal" begin # C3
        Random.seed!(0)
        traces = particle_IMH(normal_normal, (), nn_obs, 10, 1_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(traces[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end
end

end # module StaticEvaluationTests
