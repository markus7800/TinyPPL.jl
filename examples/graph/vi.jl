# Variational inference for graph-based models on a chain of normals
# (see static/vi.jl for an overview of variational families and ELBO estimators).
#
# The chain x1 → x2 → ... → x8 with x1 ~ Normal(0, 1) and x_k ~ Normal(x_{k-1} + 1, 1) is jointly Gaussian with
# mean k-1 and covariance Cov(x_i, x_j) = min(i, j), so the variables are strongly correlated.
# Graph-based models know the dependency structure, which bbvi_rao exploits for Rao-Blackwellised gradient estimates
# (Ranganath et al., Black Box Variational Inference, AISTATS 2014).
#
# Run: julia --project=examples examples/graph/vi.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import Random
import LinearAlgebra: diag, inv

model = @pgm normal_chain begin
    let x1 ~ Normal(0., 1.),
        x2 ~ Normal(x1 + 1., 1.),
        x3 ~ Normal(x2 + 1., 1.),
        x4 ~ Normal(x3 + 1., 1.),
        x5 ~ Normal(x4 + 1., 1.),
        x6 ~ Normal(x5 + 1., 1.),
        x7 ~ Normal(x6 + 1., 1.),
        x8 ~ Normal(x7 + 1., 1.)

        x8
    end
end

K = model.n_latents
exact_μ = collect(0.:K-1)
exact_Σ = [min(i, j) for i in 1:K, j in 1:K]
# The optimal mean-field Gaussian has the exact mean and the standard deviations of the conditionals p(x_k | x_{-k}),
# 1/sqrt(diag(Σ⁻¹)): 1/√2 for all but the last variable (which only depends on x7) and 1 for x8.
meanfield_std = 1 ./ sqrt.(diag(inv(exact_Σ)))
println("optimal mean-field std: ", round.(meanfield_std, digits=3))

function print_meanfield_error(method, mu, sigma)
    println(rpad(method, 24), "max. error of mean: ", round(maximum(abs, mu .- exact_μ), digits=3),
        ", of std: ", round(maximum(abs, sigma .- meanfield_std), digits=3))
end

Random.seed!(0)
vi = advi_meanfield(model, 20_000, 10, 0.05)
print_meanfield_error("ADVI mean-field", vi.Q.mu, vi.Q.sigma)

# The score function estimator needs more samples per step (L = 100) because of its higher variance.
# bbvi_naive is a hand-written implementation, bbvi uses the generic advi with ReinforceELBO.
# bbvi_rao uses for the gradient of each variable's parameters only the terms of the ELBO in its Markov blanket,
# which reduces the variance of the estimate.
Random.seed!(0)
vi = bbvi_naive(model, 10_000, 100, 0.01)
print_meanfield_error("BBVI naive", [d.base.μ for d in vi.Q.dists], [d.base.σ for d in vi.Q.dists])

Random.seed!(0)
vi = bbvi(model, 10_000, 100, 0.01)
print_meanfield_error("BBVI", [d.base.μ for d in vi.Q.dists], [d.base.σ for d in vi.Q.dists])

Random.seed!(0)
vi = bbvi_rao(model, 10_000, 100, 0.01)
print_meanfield_error("BBVI Rao-Blackwellised", [d.base.μ for d in vi.Q.dists], [d.base.σ for d in vi.Q.dists])

# Only a full-rank Gaussian can capture the correlations and the marginal variances k.
Random.seed!(0)
vi = advi_fullrank(model, 50_000, 10, 0.02)
println(rpad("ADVI full-rank", 24), "max. error of mean: ", round(maximum(abs, vi.Q.base.μ .- exact_μ), digits=3),
    ", of covariance: ", round(maximum(abs, vi.Q.base.Σ .- exact_Σ), digits=3))
println("ADVI full-rank: marginal variances ≈ ", round.(diag(vi.Q.base.Σ), digits=2), " (exact: 1:8)")
