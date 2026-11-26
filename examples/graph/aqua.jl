# AQUA: approximate inference by quantising continuous distributions
# (Huang, Dutta, Misailovic: AQUA: Automated Quantized Inference for Probabilistic Programs, ATVA 2021).
#
# AQUA splits the support of each continuous variable into N intervals, which turns a graph-based model with
# continuous variables into a discrete factor graph. Exact inference on this factor graph (variable elimination :ve,
# belief propagation :bp, or the junction tree algorithm :jt) gives approximate marginal densities.
# Belief propagation is only exact for factor graphs that are trees, otherwise use :ve or :jt.
# aqua(model, N; method) returns a dictionary from addresses to (grid points, density values).
#
# Run: julia --project=examples examples/graph/aqua.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import Distributions as Dists
import Random
using Plots
include("../common/plotting.jl")

# mean and standard deviation of a density on a uniform grid
function moments(xs, ps)
    W = ps ./ sum(ps)
    m = W'xs
    return m, sqrt(W' * (xs .- m) .^ 2)
end

# ===== A conjugate model =====
# X ~ Normal(0, 1), Y ~ Normal(X, 1), and we observe 3 ~ Normal(Y, 1).
# Exact posterior: X | 3 ~ Normal(1, sqrt(2/3)) and Y | 3 ~ Normal(2, sqrt(2/3)).
model = @pgm conjugate begin
    let X ~ Normal(0., 1.),
        Y ~ Normal(X, 1.)

        Normal(Y, 1.) ↦ 3.
    end
end

println("exact:  X ≈ 1.0 ± ", round(sqrt(2/3), digits=3), ", Y ≈ 2.0 ± ", round(sqrt(2/3), digits=3))
for method in [:ve, :bp, :jt] # the factor graph is a chain, so belief propagation works
    result = aqua(model, 200, method=method)
    println("AQUA ", rpad(method, 3), " X ≈ ", join(round.(moments(result[:X]...), digits=3), " ± "),
        ", Y ≈ ", join(round.(moments(result[:Y]...), digits=3), " ± "))
end

# ===== Distribution of the return value =====
# aqua_get_return_distribution evaluates the return expression on the quantised joint distribution of its variables.
# Here, there are no observations, so the result is the distribution of a function of independent random variables.
function return_distribution_error(model, N, exact_pdf; kwargs...)
    zs, ps = aqua_get_return_distribution(model, N, aqua(model, N, method=:ve); kwargs...)
    return zs, ps, maximum(abs, ps .- exact_pdf.(zs))
end

sum_model = @pgm sum_model begin
    let X ~ Normal(0., 1.),
        Y ~ Normal(0., 1.)
        X + Y
    end
end
zs_sum, ps_sum, err = return_distribution_error(sum_model, 100, z -> Dists.pdf(Normal(0., sqrt(2.)), z))
println("\nX + Y for independent standard normal X, Y: max. density error vs. Normal(0, √2): ", round(err, digits=4))

ratio_model = @pgm ratio_model begin
    let X ~ Normal(0., 1.),
        Y ~ Normal(0., 1.)
        X / Y
    end
end
zs_ratio, ps_ratio, err = return_distribution_error(ratio_model, 200, z -> Dists.pdf(Dists.Cauchy(), z); z0=-10., z1=10.)
println("X / Y: max. density error vs. Cauchy: ", round(err, digits=4))

mixture_model = @pgm mixture_model begin
    let X ~ Normal(-2., 1.),
        Y ~ Normal(2., 1.),
        B ~ Bernoulli(0.5)

        if B == 1.
            X
        else
            Y
        end
    end
end
mixture_pdf(z) = 0.5 * Dists.pdf(Normal(-2., 1.), z) + 0.5 * Dists.pdf(Normal(2., 1.), z)
mixture_cdf(z) = 0.5 * Dists.cdf(Normal(-2., 1.), z) + 0.5 * Dists.cdf(Normal(2., 1.), z)
zs_mix, ps_mix, err = return_distribution_error(mixture_model, 200, mixture_pdf)
_, _, err_cdf = return_distribution_error(mixture_model, 200, mixture_cdf; mode=:cdf)
println("if B then X else Y: max. density error vs. mixture: ", round(err, digits=4), ", max. cdf error: ", round(err_cdf, digits=4))

# X * Y has no elementary density, but X * Y = ((X + Y)^2 - (X - Y)^2) / 4, and (X + Y)/√2 and (X - Y)/√2 are
# independent standard normal, so X * Y has the same distribution as U/2 - V/2 for independent U, V ~ χ²(1).
# We compare the two AQUA results on the same grid. The density is infinite at 0, so the grid cannot resolve it there.
product_model = @pgm product_model begin
    let X ~ Normal(0., 1.),
        Y ~ Normal(0., 1.)
        X * Y
    end
end
chisq_model = @pgm chisq_model begin
    let U ~ Main.Dists.Chisq(1.),
        V ~ Main.Dists.Chisq(1.)
        0.5 * U - 0.5 * V
    end
end
grid = (z0=-5., z1=5.)
zs_prod, ps_prod = aqua_get_return_distribution(product_model, 100, aqua(product_model, 100, method=:ve); grid...)
_, ps_chisq = aqua_get_return_distribution(chisq_model, 100, aqua(chisq_model, 100, method=:ve); grid...)
_, cdf_prod = aqua_get_return_distribution(product_model, 100, aqua(product_model, 100, method=:ve); mode=:cdf, grid...)
_, cdf_chisq = aqua_get_return_distribution(chisq_model, 100, aqua(chisq_model, 100, method=:ve); mode=:cdf, grid...)
println("X * Y vs. U/2 - V/2: max. cdf difference: ", round(maximum(abs, cdf_prod .- cdf_chisq), digits=4),
    ", max. density difference for |z| > 0.5: ", round(maximum(abs, (ps_prod .- ps_chisq)[abs.(zs_prod) .> 0.5]), digits=4))

# For discrete variables, the return distribution is a probability mass function.
bernoulli_sum = @pgm bernoulli_sum begin
    let X ~ Bernoulli(0.2),
        Y ~ Bernoulli(0.5)
        X + Y
    end
end
result = aqua(bernoulli_sum, 100, method=:ve)
zs, pmf = aqua_get_return_distribution(bernoulli_sum, 100, result)
_, cdf = aqua_get_return_distribution(bernoulli_sum, 100, result, mode=:cdf)
println("X + Y for X ~ Bernoulli(0.2), Y ~ Bernoulli(0.5): P(X + Y = ", Int.(zs), ") = ", round.(pmf, digits=4),
    " (exact: [0.4, 0.5, 0.1]), cdf: ", round.(cdf, digits=4), " (exact: [0.4, 0.9, 1.0])")

p = plot(layout=(1, 4), size=(1300, 300), legend=false)
plot!(p[1], zs_sum, ps_sum, title="X + Y"); plot!(p[1], z -> Dists.pdf(Normal(0., sqrt(2.)), z), color=:black, linestyle=:dash)
plot!(p[2], zs_ratio, ps_ratio, title="X / Y"); plot!(p[2], z -> Dists.pdf(Dists.Cauchy(), z), color=:black, linestyle=:dash)
plot!(p[3], zs_mix, ps_mix, title="if B then X else Y"); plot!(p[3], mixture_pdf, color=:black, linestyle=:dash)
plot!(p[4], zs_prod, ps_prod, title="X * Y vs. U/2 - V/2"); plot!(p[4], zs_prod, ps_chisq, color=:black, linestyle=:dash)
show_plot(p, "graph_aqua_return_distributions")

# ===== Benchmarks from the AQUA paper =====
# Without a closed-form posterior, we compare the posterior mean of one variable with likelihood weighting.
# Grid sizes N are chosen such that each model takes a few seconds.

# Student-t observations with a bimodal prior for their location.
PriorMix = @pgm PriorMix begin
    let N = 10,
        y = [1.6024, 0.880643, -0.792556, -1.02184, -0.530578, 0.494777, -1.51075, -0.630318, -1.18208, 0.187967],

        b ~ Bernoulli(0.8),
        m = (2 * b - 1) * 4,

        mu ~ Main.Dists.truncated(Normal(m,0.5),-5.,5)

        [{:y => i} ~ TransformedDistribution(TDist(5), AffineTransform(2., mu)) ↦ y[i] for i in 1:N]

        mu
    end
end

# A linear classifier: each correctly classified point multiplies the density by e (Exponential(1/e) at 0).
ZeroOne = @pgm ZeroOne begin
    let N = 20,
        y = [1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1],
        x1 = [6, 8, -1, 0, 5, 1.2, -2, 9.8, 4, 12, 1, 10, 1, 2.2, -6, 9.8, 1, 1, 1, 1],

        w1 ~ Uniform(-10,10),
        w2 ~ Uniform(-10,10)

        [Exponential(1/exp(y[i] * (x1[i] * w1 + w2) < 0 ? 0 : 1)) ↦ 0. for i in 1:N]

        w2
    end
end

# Tug of war: each match outcome y[i] that agrees with who is stronger multiplies the density by e.
Tug = @pgm Tug begin
    let N = 40,
        y = [0,1,1,1,0,0,1,0,0,1,0,1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1],

        b_alice ~ Bernoulli(0.33),
        alice ~ Main.Dists.truncated(b_alice == 1 ? Normal(1.,1.) : Normal(0.5,0.5), -5, 5),

        b_bob ~ Bernoulli(0.33),
        bob ~ Main.Dists.truncated(b_bob == 1 ? Normal(1.,1.) : Normal(0.5,0.5), -5, 5)

        [Exponential(1/exp((abs(alice) > abs(bob)) == y[i] ? 1 : 0)) ↦ 0. for i in 1:N]

        alice
    end
end

# Only 3 mu1 mu2 - mu3 is identified by the data.
AlterMu = @pgm AlterMu begin
    let N = 40,
        y = [-2.57251482,  0.33806206,  2.71757796,  1.09861336,  2.85603752,
        -0.91651351,  0.15555127, -2.68160347,  2.47043789,  3.47459025,
        1.63949862, -1.32148757,  2.64187513,  0.30357848, -4.09546231,
        -1.50709863, -0.99517866, -2.0648892 , -2.40317949,  3.46383544,
        0.91173696,  1.18222221,  0.04235722, -0.52815171,  1.15551598,
        -1.62749724,  0.71473237, -1.08458812,  4.66020296,  1.24563831,
        -0.67970862,  0.93461681,  1.18187607, -1.49501051,  2.44755622,
        -2.06424237, -0.04584074,  1.93396696,  1.07685273, -0.09837907],

        mu1 ~ Main.Dists.truncated(Normal(0.,5.), -2, 2),
        mu2 ~ Main.Dists.truncated(Normal(0.,5.), -2, 2),
        mu3 ~ Main.Dists.truncated(Normal(0.,5.), -2, 2)

        [Normal((3.0 * mu1 * mu2) - mu3, 1.) ↦ y[i] for i in 1:N]

        mu1
    end
end

# Only mu1 + mu2 is identified by the data.
AlterMu2 = @pgm AlterMu2 begin
    let N = 40,
        y = [-2.57251482,  0.33806206,  2.71757796,  1.09861336,  2.85603752,
        -0.91651351,  0.15555127, -2.68160347,  2.47043789,  3.47459025,
        1.63949862, -1.32148757,  2.64187513,  0.30357848, -4.09546231,
        -1.50709863, -0.99517866, -2.0648892 , -2.40317949,  3.46383544,
        0.91173696,  1.18222221,  0.04235722, -0.52815171,  1.15551598,
        -1.62749724,  0.71473237, -1.08458812,  4.66020296,  1.24563831,
        -0.67970862,  0.93461681,  1.18187607, -1.49501051,  2.44755622,
        -2.06424237, -0.04584074,  1.93396696,  1.07685273, -0.09837907],

        mu1 ~ Uniform(-10,10),
        mu2 ~ Uniform(-10,10)

        [Normal(mu1 + mu2, 1.) ↦ y[i] for i in 1:N]

        mu1
    end
end

# Robust estimation of a mean a, with a separate precision for each observation.
AnovaRP = @pgm AnovaRP begin
    let N = 40,
        y = [0.617007250574049, 0.895262019894671, -0.729874754380836,
        0.301471152630035, 0.929871984146596, 1.70686064033886, 1.09782179678289,
        1.27082634042846, 1.75132331738128, 1.03490015378005, 1.71226194303743,
        0.432311329644381, 1.04451460234109, 0.797605506723309, 1.19837496179441,
        2.00164115453824, -0.377675819779764, -0.148489586942737, 2.12586836175586,
        0.560257670828821, 0.751966977556745, 0.346571453512205, 1.2187492301484,
        1.56886622653154, 1.38035019540838, 2.76837231107256, 0.323735749142999,
        1.42941129934677, 0.468406777114355, 0.751905484130352, 0.568364441627544,
        0.0139444580475871, -0.152744699487943, 1.89057571718701, 1.48479996615007,
        1.87341593940818, 1.89344977488441, 0.537059334676276, 0.944371774385459,
        -0.474992028001801],

        a ~ Uniform(0., 3.),
        mu1 ~ Uniform(-10,10),
        sigma_y ~ Uniform(0.,100),
        robust_local_tau = [{:tau => i} ~ Uniform(0, 10) for i in 1:N]

        [Normal(a, sigma_y / sqrt(max(1e-5,robust_local_tau[i]))) ↦ y[i] for i in 1:N]

        a
    end
end

function lw_mean(model, addr)
    Random.seed!(0)
    traces, logw = likelihood_weighting(model, 1_000_000)
    return exp.(logw)'traces[addr]
end

println()
for (model, addr, N) in [(PriorMix, :mu, 1000), (ZeroOne, :w2, 200), (Tug, :alice, 60), (AlterMu, :mu1, 30), (AlterMu2, :mu1, 200), (AnovaRP, :a, 30)]
    result = aqua(model, N, method=:ve)
    m, _ = moments(result[addr]...)
    println(rpad(string(model.name), 9), " E[", addr, "]: AQUA (N = ", N, ") ≈ ", round(m, digits=3), ", likelihood weighting ≈ ", round(lw_mean(model, addr), digits=3))
end
