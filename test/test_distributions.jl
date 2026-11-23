module DistributionsTests
# Unit tests for src/distributions and src/logjoint. Mostly deterministic.

using Test, Random, Statistics
import ForwardDiff, ReverseDiff, Tracker
import Distributions
import Distributions: Normal, Gamma, Beta, Uniform, Exponential, InverseGamma, Laplace, LogNormal, Cauchy,
    Poisson, Geometric, Bernoulli, Categorical, Binomial, DiscreteUniform, MvNormal, RealInterval, TDist, logpdf, pdf
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
            # A5: domain and image (also of compositions with decreasing transforms)
            @test TD.domain(T) == interval
            @test TD.domain(T_inv) == RealInterval(-Inf, Inf)
            @test TD.image(T_inv, RealInterval(-Inf, Inf)) == interval
            if interval != RealInterval(2.0, 5.0) # the affine map rounds 5 to 1 - eps
                @test TD.image(T, interval) == RealInterval(-Inf, Inf)
            end
        end
    end
    # support of a transformed distribution with decreasing transform x ↦ log(1 - x)
    @test Distributions.support(TD.to_unconstrained(Distributions.truncated(Normal(0.0, 1.0), -Inf, 1.0))) == RealInterval(-Inf, Inf)

    @testset "sigmoid jacobian does not overflow" begin # J5
        T = TD.SigmoidTransform()
        @test isfinite(TD.log_abs_det_jacobian(T, 800.0))
        @test TD.log_abs_det_jacobian(T, 800.0) ≈ -800.0
        @test TD.log_abs_det_jacobian(T, -800.0) ≈ -800.0
        for x in (-3.0, -0.5, 0.0, 0.7, 4.0)
            @test TD.log_abs_det_jacobian(T, x) ≈ log(ForwardDiff.derivative(T, x))
            # derivative at 0 is 0 (symmetric), although written with abs
            @test ForwardDiff.derivative(y -> TD.log_abs_det_jacobian(T, y), x) ≈ (1 - 2T(x))
        end
        # inverse: log |d/dy σ⁻¹(y)| = -log σ'(σ⁻¹(y)), also close to 1
        T_inv = TD.InverseSigmoidTransform()
        for y in (1e-8, 0.2, 0.5, 0.9, 1 - 1e-12)
            @test TD.log_abs_det_jacobian(T_inv, y) ≈ -TD.log_abs_det_jacobian(T, T_inv(y))
        end
        @test TD.log_abs_det_jacobian(T_inv, 1 - 1e-12) ≈ -log(1e-12) rtol = 1e-6
    end
end

@testset "transformed distributions are concretely typed" begin
    for d in [Gamma(2.0, 1.0), Uniform(2.0, 5.0), Distributions.truncated(Normal(0.0, 1.0), -Inf, 1.0), Beta(2.0, 3.0)]
        d_unconstrained = TD.to_unconstrained(d)
        @test all(isconcretetype, fieldtypes(typeof(d_unconstrained)))
        @test (@inferred logpdf(d_unconstrained, 0.3)) isa Float64
        @test (@inferred d_unconstrained.T_inv(0.3)) isa Float64
        @test (@inferred rand(d_unconstrained)) isa Float64
    end
    @test TD.AffineTransform(1, 2.0) isa TD.AffineTransform{Float64} # promoted
end

@testset "to_unconstrained densities integrate to 1" begin
    grid = range(-40, 40, length=200_001)
    for d in [Gamma(2.0, 1.0), Exponential(2.0), Beta(2.0, 3.0), Uniform(2.0, 5.0), LogNormal(0.0, 1.0), InverseGamma(3.0, 2.0),
              Distributions.truncated(Normal(0.0, 1.0), -Inf, 1.0)] # B1: upper-bounded support
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
    @testset "proposals are concretely typed" begin
        for q in [TD.StaticProposal(Normal(0., 1.)), TD.ContinuousRandomWalkProposal(1), TD.ContinuousRandomWalkProposal(0.5, 0, 3),
                  TD.DiscreteRandomWalkProposal(2), TD.DiscreteRandomWalkProposal(2., 1, 6),
                  TD.random_walk_proposal_dist(Binomial(20, 0.3), 5, 2.), TD.random_walk_proposal_dist(Poisson(3.), 5, 2.)]
            @test all(isconcretetype, fieldtypes(typeof(q)))
        end
        d = TD.random_walk_proposal_dist(Poisson(3.), 5, 2.)
        @test (@inferred logpdf(d, 4)) isa Float64
        @test (@inferred rand(d)) isa Int
        # bounded discrete proposal stays in the support
        q = TD.DiscreteRandomWalkProposal(2., 1, 6)
        @test all(1 <= TD.propose_and_logpdf(q, 3)[1] <= 6 for _ in 1:1000)
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
    @test Distributions.entropy(q) ≈ Distributions.entropy(MvNormal([0.5, -1.0, 2.0], L_neg * L_neg'))

    # J5: product of the diagonal of L underflows in high dimensions
    K_high = 400
    q = TD.FullRankGaussian(zeros(K_high), vec(Matrix(0.1 * TD.LinearAlgebra.I(K_high))))
    @test Distributions.entropy(q) ≈ K_high/2 * (log(2π) + 1) + K_high * log(0.1)

    @testset "FullRankGaussian gradients with negative diagonal" begin # B16
        q0 = TD.FullRankGaussian(3)
        phi = vcat([0.5, -1.0, 2.0], vec(L_neg))
        f(p) = (q_p = TD.update_params(q0, p); logpdf(q_p, x) + Distributions.entropy(q_p))
        h = 1e-6
        ref = [(f(phi .+ h .* (1:12 .== i)) - f(phi .- h .* (1:12 .== i))) / 2h for i in 1:12]
        @test Tracker.data(Tracker.gradient(f, phi)[1]) ≈ ref rtol = 1e-5
        @test ForwardDiff.gradient(f, phi) ≈ ref rtol = 1e-5
        @test ReverseDiff.gradient(f, phi) ≈ ref rtol = 1e-5
    end

    @testset "rand_reparam" begin # J6
        # same random numbers as Distributions.rand for Float64 parameters
        for d in [Normal(0.5, 2.0), LogNormal(0.1, 0.5), MvNormal([0.5, -1.0, 2.0], L * L'), Gamma(2.0, 1.0)]
            Random.seed!(3); a = TD.rand_reparam(d)
            Random.seed!(3); b = rand(d)
            @test a ≈ b
        end
        # reparametrised samples are differentiable with all AD backends (ReverseDiff has no randn(TrackedReal))
        q0 = TD.FullRankGaussian(3)
        phi = vcat([0.5, -1.0, 2.0], vec(L))
        function g(p)
            Random.seed!(1)
            z, lz = TD.rand_and_logpdf(TD.update_params(q0, p))
            return sum(abs2, z) + lz
        end
        ref = ForwardDiff.gradient(g, phi)
        @test Tracker.data(Tracker.gradient(g, phi)[1]) ≈ ref
        @test ReverseDiff.gradient(g, phi) ≈ ref
        n(p) = (Random.seed!(1); TD.rand_reparam(Normal(p[1], p[2])))
        @test ReverseDiff.gradient(n, [0.5, 2.0]) ≈ ForwardDiff.gradient(n, [0.5, 2.0])
    end

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
        (θ -> Poisson(θ[1]), [3.0], 2.0), # values stored as Float64 in traces
        (θ -> Geometric(θ[1]), [0.3], 2),
        (θ -> Bernoulli(θ[1]), [0.3], true),
        (θ -> Bernoulli(θ[1]), [0.3], false),
        (θ -> Bernoulli(θ[1]), [0.3], 1.0),
        (θ -> TDist(θ[1]), [3.5], 1.1),
        (θ -> Uniform(θ[1], θ[2]), [-0.5, 2.0], 1.1),
        (θ -> Categorical(θ), [0.2, 0.3, 0.5], 2),
        (θ -> Exponential(θ[1]), [2.0], 1), # Int value for continuous distribution
    ]
    for (make_dist, θ, x) in cases
        d = make_dist(θ)
        @testset "$(typeof(d).name.name) x=$x" begin
            ad_grad = ForwardDiff.gradient(θ -> logpdf(make_dist(θ), x), θ)
            @test collect(Iterators.flatten(TD.logpdf_params_grad(d, x))) ≈ ad_grad
        end
    end
    @testset "Binomial" begin # n is discrete, gradient only w.r.t. p
        ∇n, ∇p = TD.logpdf_params_grad(Binomial(5, 0.3), 2)
        @test ∇n == 0
        @test ∇p ≈ ForwardDiff.derivative(p -> logpdf(Binomial(5, p), 2), 0.3)
    end
    @testset "constrain" begin
        @test TD.constrain(Bernoulli(0.5), (1.3,)) == (1.,)
        @test TD.constrain(Binomial(5, 0.5), (5, -0.2)) == (5, 0.)
        @test_throws ErrorException TD.constrain(Normal(), (0., 1.))
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


import TinyPPL
@testset "no type piracy of randn, reshape, pop!" begin # J6
    for f in (randn, reshape, pop!)
        @test !any(m -> Base.moduleroot(m.module) === TinyPPL, methods(f))
    end
end

end # module DistributionsTests
