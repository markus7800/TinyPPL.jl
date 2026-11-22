module DistributionsTests
# Unit tests for src/distributions and src/logjoint. Mostly deterministic.

using Test, Random, Statistics
import ForwardDiff
import Distributions
import Distributions: Normal, Gamma, Beta, Uniform, Exponential, InverseGamma, Laplace, LogNormal, Cauchy,
    Poisson, Geometric, Bernoulli, Categorical, Binomial, DiscreteUniform, MvNormal, RealInterval, logpdf, pdf
import TinyPPL.Distributions as TD
import TinyPPL.Logjoint as LJ
using ..TestUtils

# trapezoidal integral of exp(logf) over a grid
function integrate_exp(logf, grid)
    ys = exp.(logf.(grid))
    return sum((ys[1:end-1] .+ ys[2:end]) ./ 2 .* diff(grid))
end

@testset "transforms" begin
    # (interval, points inside the interval)
    cases = [
        (RealInterval(-Inf, Inf), [-3.0, 0.0, 2.5]),
        (RealInterval(0.0, Inf), [1e-3, 1.0, 10.0]),
        (RealInterval(2.0, Inf), [2.001, 3.0, 50.0]),
        (RealInterval(-Inf, 1.0), [-10.0, 0.0, 0.999]), # B1
        (RealInterval(0.0, 1.0), [1e-3, 0.5, 0.999]),
        (RealInterval(2.0, 5.0), [2.001, 3.5, 4.999]),
    ]
    for (interval, xs) in cases
        @testset "$interval" begin
            T = TD.transform_to_unconstrained_from(interval)
            T_inv = TD.transform_to(interval)
            for x in xs
                y = T(x)
                @test isfinite(y)
                @test T_inv(y) ≈ x
                # log |dT/dx|
                @test TD.log_abs_det_jacobian(T, x) ≈ log(abs(ForwardDiff.derivative(T, x)))
                @test TD.log_abs_det_jacobian(T_inv, y) ≈ log(abs(ForwardDiff.derivative(T_inv, y)))
            end
        end
    end

    @testset "sigmoid jacobian does not overflow" begin # J5
        T = TD.SigmoidTransform()
        @test isfinite(TD.log_abs_det_jacobian(T, 800.0))
    end
end

@testset "to_unconstrained densities integrate to 1" begin
    grid = range(-40, 40, length=200_001)
    for d in [Gamma(2.0, 1.0), Exponential(2.0), Beta(2.0, 3.0), Uniform(2.0, 5.0), LogNormal(0.0, 1.0), InverseGamma(3.0, 2.0)]
        d_unconstrained = TD.to_unconstrained(d)
        @test integrate_exp(x -> logpdf(d_unconstrained, x), grid) ≈ 1 atol = 1e-4
    end
end

@testset "random walk proposals" begin
    @testset "discrete proposals are normalised and match sampling" begin
        Random.seed!(0)
        for (d, x) in [(Categorical([0.2, 0.3, 0.5]), 2), (Categorical([0.2, 0.3, 0.5]), 1),
                       (DiscreteUniform(-2, 4), 4), (Binomial(5, 0.3), 0), (Poisson(3.0), 0), (Poisson(3.0), 7)]
            q = TD.random_walk_proposal_dist(d, x, 1.0)
            support = -5:60
            ps = [exp(logpdf(q, k)) for k in support]
            @test sum(ps) ≈ 1 atol = 1e-8
            @test all(ps[support .== x] .== 0) # never proposes current value
            samples = [rand(q) for _ in 1:100_000]
            @test frequencies(samples, support) ≈ ps atol = 0.01
        end
    end

    @testset "single category proposal does not crash" begin # C15
        q = TD.random_walk_proposal_dist(Categorical([1.0]), 1, 1.0)
        @test rand(q) == 1 || logpdf(q, rand(q)) == -Inf
    end

    @testset "continuous bounded proposals are normalised and match sampling" begin
        Random.seed!(0)
        # x close to the bounds is tested on purpose (C15)
        for (d, x) in [(Beta(2.0, 2.0), 0.5), (Beta(2.0, 2.0), 1e-4), (Beta(2.0, 2.0), 1 - 1e-4),
                       (Gamma(2.0, 1.0), 3.0), (Gamma(2.0, 1.0), 1e-4), (Uniform(-1.0, 2.0), 1.9999)]
            q = TD.random_walk_proposal_dist(d, x, 0.5)
            lb, ub = max(Distributions.minimum(d), -50.0), min(Distributions.maximum(d), 50.0)
            grid = range(lb, ub, length=400_001)
            @test integrate_exp(y -> logpdf(q, y), grid) ≈ 1 atol = 1e-3
            # the density must describe the sampler: compare P(Y < x) under both
            samples = [rand(q) for _ in 1:100_000]
            @test all(lb .<= samples .<= ub)
            p_left_sampled = mean(samples .< x)
            p_left_density = integrate_exp(y -> logpdf(q, y), range(lb, x, length=200_001))
            @test p_left_sampled ≈ p_left_density atol = 0.01
        end
    end
end

@testset "variational distributions" begin
    Random.seed!(0)
    K = 3
    q = TD.update_params(TD.MeanFieldGaussian(K), [0.5, -1.0, 2.0, log(0.5), log(2.0), log(1.0)])
    ref = MvNormal([0.5, -1.0, 2.0], [0.5, 2.0, 1.0])
    x = [0.1, 0.2, 0.3]
    @test logpdf(q, x) ≈ logpdf(ref, x)
    @test Distributions.entropy(q) ≈ Distributions.entropy(ref)
    z, lz = TD.rand_and_logpdf(q)
    @test lz ≈ logpdf(ref, z)

    L = [1.0 0.0 0.0; 0.5 2.0 0.0; -0.3 0.2 0.7]
    q = TD.FullRankGaussian([0.5, -1.0, 2.0], vec(L))
    ref = MvNormal([0.5, -1.0, 2.0], L * L')
    @test logpdf(q, x) ≈ logpdf(ref, x)
    @test Distributions.entropy(q) ≈ Distributions.entropy(ref)

    # negative diagonal entries of L are a valid parametrisation of the same covariance
    L_neg = [-1.0 0.0 0.0; 0.5 2.0 0.0; -0.3 0.2 0.7]
    q = TD.FullRankGaussian([0.5, -1.0, 2.0], vec(L_neg))
    @test logpdf(q, x) ≈ logpdf(MvNormal([0.5, -1.0, 2.0], L_neg * L_neg'), x)

    @testset "MeanField sampling returns Float64 matrix" begin # B8
        q = TD.MeanField([TD.VariationalNormal(), TD.VariationalNormal()])
        @test rand(q, 10) isa Matrix{Float64}
    end
end

@testset "logpdf_params_grad matches AD" begin # B9
    cases = [
        (θ -> Normal(θ[1], θ[2]), [0.3, 1.7], 1.1),
        (θ -> Exponential(θ[1]), [2.0], 0.7),
        (θ -> Gamma(θ[1], θ[2]), [2.0, 1.5], 0.7),
        (θ -> InverseGamma(θ[1], θ[2]), [3.0, 2.0], 0.7),
        (θ -> Beta(θ[1], θ[2]), [2.0, 3.0], 0.3),
        (θ -> Laplace(θ[1], θ[2]), [0.2, 1.5], 1.1),
        (θ -> LogNormal(θ[1], θ[2]), [0.2, 1.5], 1.1),
        (θ -> Cauchy(θ[1], θ[2]), [0.2, 1.5], 1.1),
        (θ -> Poisson(θ[1]), [3.0], 2),
        (θ -> Geometric(θ[1]), [0.3], 2),
        (θ -> Bernoulli(θ[1]), [0.3], true),
    ]
    for (make_dist, θ, x) in cases
        d = make_dist(θ)
        @testset "$(typeof(d).name.name)" begin
            ad_grad = ForwardDiff.gradient(θ -> logpdf(make_dist(θ), x), θ)
            @test collect(TD.logpdf_params_grad(d, x)) ≈ ad_grad
        end
    end
end

@testset "ELBO gradient AD backends agree" begin # B2
    for K in (2, 8) # 4 and 16 variational parameters
        logjoint(x) = -sum(abs2, x .- 3.0) / 2
        q = TD.MeanFieldGaussian(K)
        phi = collect(range(-1, 1, length=2K))
        Random.seed!(1); g_tracker = LJ.estimate_elbo_grad_tracker(logjoint, phi, q, 1, TD.RelativeEntropyELBO())
        Random.seed!(1); g_forward = LJ.estimate_elbo_grad_forwarddiff(logjoint, phi, q, 1, TD.RelativeEntropyELBO())
        Random.seed!(1); g_reverse = LJ.estimate_elbo_grad_reversediff(logjoint, phi, q, 1, TD.RelativeEntropyELBO())
        @test length(g_forward) == 2K
        @test collect(g_forward) ≈ collect(g_tracker)
        @test collect(g_reverse) ≈ collect(g_tracker)
    end
end

@testset "logjoint HMC and ADVI on a Gaussian" begin
    # target N([1, -2], diag([0.5, 2]^2))
    μ, σ = [1.0, -2.0], [0.5, 2.0]
    logjoint(x) = -sum(abs2, (x .- μ) ./ σ) / 2
    for backend in (:tracker, :forwarddiff, :reversediff)
        @testset "hmc $backend" begin
            Random.seed!(0)
            samples = LJ.hmc_logjoint(logjoint, 2, 4_000, 10, 0.1; ad_backend=backend)
            @test vec(mean(samples, dims=2)) ≈ μ atol = 0.2
            @test vec(std(samples, dims=2)) ≈ σ atol = 0.2
        end
    end
    @testset "advi_meanfield_logjoint" begin
        Random.seed!(0)
        q = LJ.advi_meanfield_logjoint(logjoint, 2, 3_000, 10, 0.05)
        @test q.mu ≈ μ atol = 0.1
        @test q.sigma ≈ σ atol = 0.15
    end
    @testset "advi_fullrank_logjoint" begin
        Random.seed!(0)
        q = LJ.advi_fullrank_logjoint(logjoint, 2, 3_000, 10, 0.05)
        @test Distributions.mean(q.base) ≈ μ atol = 0.1
        @test sqrt.(Distributions.var(q.base)) ≈ σ atol = 0.15
    end
    for estimator in (TD.MonteCarloELBO(), TD.RelativeEntropyELBO(), TD.PathDerivativeELBO(), TD.ReinforceELBO())
        @testset "advi_logjoint $(typeof(estimator))" begin
            Random.seed!(0)
            q = LJ.advi_logjoint(logjoint, 3_000, 10, 0.05, TD.MeanFieldGaussian(2), estimator)
            @test q.mu ≈ μ atol = 0.15
            @test q.sigma ≈ σ atol = 0.2
        end
    end
end

end # module DistributionsTests
