# Variational inference for universal models.
#
# For universal models, the variational distribution Q is a mean-field distribution over addresses,
# which is extended whenever a run of the model instantiates a new address.
# This example shows that the universal implementations give the same results as the static ones on
# linear regression (see static/vi.jl for an overview of variational families and ELBO estimators),
# and applies BBVI to a model with a random number of variables.
# See also discontinuous_vi.jl for VI with a custom guide.
#
# Run: julia --project=examples examples/universal/vi.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots
include("../common/linear_regression.jl")
include("../common/plotting.jl")

# ===== Linear regression =====
@ppl function linear_regression(xs)
    intercept ~ Normal(0., prior_σ)
    slope ~ Normal(0., prior_σ)
    for i in eachindex(xs)
        {:y => i} ~ Normal(intercept + slope * xs[i], σ)
    end
    return (intercept, slope)
end

args = (xs,)
observations = Observations((:y => i) => y for (i, y) in enumerate(ys))

print_posterior("exact", posterior_μ, posterior_std)
print_posterior("optimal mean-field", posterior_μ, 1 ./ sqrt.(diag(inv(posterior_Σ))))

# Q is indexed by address. The result is the same as for the static model in static/vi.jl.
Random.seed!(0)
vi = advi_meanfield(linear_regression, args, observations, 10_000, 10, 0.01)
print_posterior("ADVI mean-field", [vi.Q[:intercept].base.μ, vi.Q[:slope].base.μ], [vi.Q[:intercept].base.σ, vi.Q[:slope].base.σ])

# A custom guide works the same way as for static models.
@ppl function guide()
    mu_intercept = param("mu_intercept")
    sigma_intercept = param("sigma_intercept", constraint=Positive())
    mu_slope = param("mu_slope")
    sigma_slope = param("sigma_slope", constraint=Positive())

    intercept ~ Normal(mu_intercept, sigma_intercept)
    slope ~ Normal(mu_slope, sigma_slope)
end

Random.seed!(0)
vi = advi(linear_regression, args, observations, 10_000, 10, 0.01, guide, (), MonteCarloELBO())
params = get_constrained_parameters(vi.Q)
print_posterior("guide", [params["mu_intercept"], params["mu_slope"]], [params["sigma_intercept"], params["sigma_slope"]])


# ===== Random number of variables =====
# A random walk with a geometrically distributed number of steps n+1, each uniform in [x-1, x+1].
# Nothing is observed, so the posterior is the prior, which lets us check the fit exactly.
@ppl function random_walk()
    n ~ Geometric(0.3)
    x = 0.
    for i in 0:Int(n)
        x = {:x => i} ~ Uniform(x - 1., x + 1.)
    end
    return x
end

# BBVI fits a distribution of each variable's prior family (here Geometric for n and Uniform for the steps),
# so it can handle the discrete n. The score function gradient estimator needs many samples per step (L = 100).
Random.seed!(0)
vi = bbvi(random_walk, (), Observations(), 10_000, 100, 0.01)
println("\nBBVI: Q(n) = ", vi.Q[:n], " (exact: Geometric(0.3)), fitted distributions for ", length(vi.Q.variational_dists), " addresses")

# Samples from Q have different addresses. Addresses that are not instantiated in a sample are `missing`.
Random.seed!(0)
traces, _ = sample_posterior(vi, 100_000)
println("BBVI: P(:x => 5 is instantiated) ≈ ", mean(.!ismissing.(traces[:x => 5])), " (exact: P(n ≥ 5) = ", round(0.7^5, digits=4), ")")

ks = 0:10
p = bar(collect(ks), [mean(traces[:n] .== k) for k in ks], label="BBVI", xlabel="n", ylabel="P(n)", xticks=ks, linewidth=0, alpha=0.6)
scatter!(p, collect(ks), exp.(logpdf.(Geometric(0.3), ks)), label="exact", color=:black)
show_plot(p, "universal_vi_random_walk")
