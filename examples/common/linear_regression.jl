# Data and exact posterior for the Bayesian linear regression examples
# (static/linear_regression.jl, graph/linear_regression.jl, */vi.jl).
#
#   intercept ~ Normal(0, prior_σ), slope ~ Normal(0, prior_σ), y_i ~ Normal(intercept + slope * x_i, σ)
#
# The prior is conjugate, so the posterior is Gaussian and can be computed in closed form.
# The xs are not centred, so intercept and slope are strongly (negatively) correlated a posteriori.

import Distributions: MvNormal, logpdf
import LinearAlgebra: I, diag, diagm

const xs = [0., 0.5, 1., 1.5, 2.]
const ys = [-3.2, -1.8, -0.5, -0.2, 1.5]
const prior_σ = 3.
const σ = 2.

const Phi = hcat(ones(length(xs)), xs) # design matrix, columns: intercept, slope
const posterior_Σ = inv(I / prior_σ^2 + Phi'Phi / σ^2)
const posterior_μ = posterior_Σ * Phi'ys / σ^2
const posterior_std = sqrt.(diag(posterior_Σ))
const log_evidence = logpdf(MvNormal(zeros(length(xs)), prior_σ^2 * Phi * Phi' + σ^2 * I), ys)

_fmt(x) = lpad(string(round(x, digits=3)), 6)
function print_posterior(method::String, μ::AbstractVector{<:Real}, std::AbstractVector{<:Real})
    println(rpad(method, 30), "intercept = ", _fmt(μ[1]), " ± ", _fmt(std[1]), "   slope = ", _fmt(μ[2]), " ± ", _fmt(std[2]))
end
# mean ± standard deviation of (weighted) samples
function print_posterior(method::String, traces, W=nothing)
    samples = hcat(traces[:intercept], traces[:slope])
    W = isnothing(W) ? fill(1 / size(samples, 1), size(samples, 1)) : W
    μ = samples'W
    std = sqrt.(((samples .- μ') .^ 2)'W)
    print_posterior(method, μ, std)
end
