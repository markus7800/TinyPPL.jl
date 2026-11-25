# Variational inference on Bayesian linear regression (see linear_regression.jl for sampling-based inference).
#
# VI fits a distribution Q to the posterior by maximising the ELBO with stochastic gradient ascent.
# advi(model, args, observations, n_samples, L, learning_rate, Q, estimator) runs n_samples gradient steps,
# each estimating the ELBO gradient with L samples from Q. Here we compare
#   - variational families: mean-field Gaussian, full-rank Gaussian, and a custom guide program,
#   - ELBO gradient estimators: reparameterisation (RelativeEntropyELBO, MonteCarloELBO, PathDerivativeELBO)
#     and score function (ReinforceELBO, as in BBVI).
# References: Kucukelbir et al., Automatic Differentiation Variational Inference (JMLR 2017);
# Ranganath et al., Black Box Variational Inference (AISTATS 2014);
# Roeder et al., Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference (NeurIPS 2017).
#
# Run: julia --project=examples examples/static/vi.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots
include("../common/linear_regression.jl")
include("../common/plotting.jl")

@ppl static function linear_regression(xs)
    intercept ~ Normal(0., prior_σ)
    slope ~ Normal(0., prior_σ)
    for i in eachindex(xs)
        {:y => i} ~ Normal(intercept + slope * xs[i], σ)
    end
    return (intercept, slope)
end

args = (xs,)
observations = Observations((:y => i) => y for (i, y) in enumerate(ys))
K = get_number_of_latent_variables(linear_regression, args, observations)

# ===== Mean-field vs. full-rank =====
# A mean-field Gaussian has independent components and cannot capture the correlation between intercept and slope.
# Its optimum matches the posterior mean, but the standard deviations are those of the conditionals p(intercept | slope, y)
# and p(slope | intercept, y), i.e. 1/sqrt(diag(Σ⁻¹)), which underestimates the uncertainty.
meanfield_std = 1 ./ sqrt.(diag(inv(posterior_Σ)))
print_posterior("exact", posterior_μ, posterior_std)
print_posterior("optimal mean-field", posterior_μ, meanfield_std)

Random.seed!(0)
vi_meanfield = advi_meanfield(linear_regression, args, observations, 10_000, 10, 0.01)
print_posterior("ADVI mean-field", vi_meanfield.Q.mu, vi_meanfield.Q.sigma)

Random.seed!(0)
vi_fullrank = advi_fullrank(linear_regression, args, observations, 10_000, 10, 0.01)
print_posterior("ADVI full-rank", vi_fullrank.Q.base.μ, sqrt.(diag(vi_fullrank.Q.base.Σ)))
println("ADVI full-rank: posterior correlation ≈ ", round(vi_fullrank.Q.base.Σ[1,2] / prod(sqrt.(diag(vi_fullrank.Q.base.Σ))), digits=3),
    " (exact: ", round(posterior_Σ[1,2] / prod(posterior_std), digits=3), ")")

# sample_posterior draws from Q and maps the samples back to the model's addresses (and support).
traces = sample_posterior(vi_fullrank, 100_000)
print_posterior("samples from full-rank Q", traces)

p = plot(xlabel="intercept", ylabel="slope", aspect_ratio=:equal)
covellipse!(p, vi_meanfield.Q.mu, diagm(vi_meanfield.Q.sigma .^ 2), n_std=2, label="mean-field", fillalpha=0.3)
covellipse!(p, vi_fullrank.Q.base.μ, vi_fullrank.Q.base.Σ, n_std=2, label="full-rank", fillalpha=0.3)
covellipse!(p, posterior_μ, posterior_Σ, n_std=2, label="exact posterior", fillalpha=0, linecolor=:black, linewidth=2, linestyle=:dash)
show_plot(p, "static_vi_meanfield_vs_fullrank")


# ===== ELBO gradient estimators =====
# advi_fullrank is advi with a FullRankGaussian and the RelativeEntropyELBO estimator,
# which computes the entropy of Q in closed form. `unconstrained=true` fits Q to the model transformed to ℝᴷ.
# We report the distance of the fitted Q to the exact posterior.
function print_fullrank_error(method, Q)
    println(rpad(method, 30), "max. error of mean: ", round(maximum(abs, Q.base.μ .- posterior_μ), digits=4),
        ", of covariance: ", round(maximum(abs, Q.base.Σ .- posterior_Σ), digits=4))
end
println()
print_fullrank_error("advi_fullrank", vi_fullrank.Q)

Random.seed!(0)
vi = advi(linear_regression, args, observations, 10_000, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO(), unconstrained=true)
print_fullrank_error("RelativeEntropyELBO", vi.Q)

# MonteCarloELBO also estimates the entropy term by sampling.
# For Gaussian Q its gradient is the same as for RelativeEntropyELBO, since the gradient of log q(t(ζ, ϕ) | ϕ)
# w.r.t. ϕ for the reparameterised sample t(ζ, ϕ) = μ + Lζ does not depend on ζ, so the results are identical.
Random.seed!(0)
vi = advi(linear_regression, args, observations, 10_000, 10, 0.01, FullRankGaussian(K), MonteCarloELBO(), unconstrained=true)
print_fullrank_error("MonteCarloELBO", vi.Q)

# PathDerivativeELBO drops the score term of the gradient, which has expectation zero.
# If Q can match the posterior exactly, as here, the variance of the gradient vanishes at the optimum,
# so the fit gets much closer (Roeder et al.). This does not help for a mean-field Q, which cannot match the posterior.
Random.seed!(0)
vi = advi(linear_regression, args, observations, 10_000, 10, 0.01, FullRankGaussian(K), PathDerivativeELBO(), unconstrained=true)
print_fullrank_error("PathDerivativeELBO", vi.Q)

# The score function estimator (ReinforceELBO) only needs log q to be differentiable, not the model,
# so it also works for discrete variables and discontinuous densities (see universal/discontinuous_vi.jl),
# but has much higher variance. bbvi uses it with a mean-field distribution of each variable's prior family,
# so we compare it to advi_meanfield and the optimal mean-field solution.
function print_meanfield_error(method, mu, sigma)
    println(rpad(method, 30), "max. error of mean: ", round(maximum(abs, mu .- posterior_μ), digits=4),
        ", of std: ", round(maximum(abs, sigma .- meanfield_std), digits=4))
end
println()
print_meanfield_error("advi_meanfield", vi_meanfield.Q.mu, vi_meanfield.Q.sigma)

Random.seed!(0)
vi = bbvi(linear_regression, args, observations, 10_000, 10, 0.01)
print_meanfield_error("BBVI (ReinforceELBO)", [d.base.μ for d in vi.Q.dists], [d.base.σ for d in vi.Q.dists])


# ===== Custom guide =====
# A guide is a program with the same latent addresses as the model, whose distributions depend on
# variational parameters declared with `param`. Constraints keep parameters in their domain (here σ > 0).
# This guide is the mean-field Gaussian, so it gives the same result as advi_meanfield (the entropy of a guide is not
# known in closed form, so we use MonteCarloELBO, which is equivalent here).
# get_constrained_parameters returns the fitted parameters by name.
@ppl static function guide()
    mu_intercept = param("mu_intercept")
    sigma_intercept = param("sigma_intercept", constraint=Positive())
    mu_slope = param("mu_slope")
    sigma_slope = param("sigma_slope", constraint=Positive())

    intercept ~ Normal(mu_intercept, sigma_intercept)
    slope ~ Normal(mu_slope, sigma_slope)
end

Random.seed!(0)
vi_guide = advi(linear_regression, args, observations, 10_000, 10, 0.01, guide, (), MonteCarloELBO(), unconstrained=true)
params = get_constrained_parameters(vi_guide.Q)
print_meanfield_error("guide", [params["mu_intercept"], params["mu_slope"]], [params["sigma_intercept"], params["sigma_slope"]])
