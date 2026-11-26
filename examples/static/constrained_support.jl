# Variables whose support depends on other variables.
#
# x ~ Uniform(-1, 1), y ~ Uniform(x - 1, x + 1), z ~ Uniform(y - 1, y + 1). Nothing is observed, so the posterior is
# the prior: y is the sum of two and z of three independent Uniform(-1, 1), i.e. y has the triangular density
# (2 - |y|) / 4 on [-2, 2], std(y) = sqrt(2/3), std(z) = 1, and P(|y| < 1) = 0.75.
#
# Gradient-based algorithms (HMC, ADVI) work in unconstrained space ℝᴷ. Each variable is mapped from its support
# to ℝ with a bijection that depends on the current values of the other variables, e.g. y ↦ logit((y - (x - 1)) / 2).
# Here, the transformed variables are even independent (each is Logistic distributed).
# See graph/constrained_support.jl for the same model as graph-based model.
#
# Run: julia --project=examples examples/static/constrained_support.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots
include("../common/plotting.jl")

@ppl static function uniform_chain()
    x ~ Uniform(-1., 1.)
    y ~ Uniform(x - 1., x + 1.)
    z ~ Uniform(y - 1., y + 1.)
end

args = ()
observations = Observations()

_std(v) = sqrt(mean((v .- mean(v)) .^ 2))
print_summary(method, traces) = println(rpad(method, 22), "std(y) ≈ ", round(_std(traces[:y]), digits=3),
    ", std(z) ≈ ", round(_std(traces[:z]), digits=3), ", P(|y| < 1) ≈ ", round(mean(abs.(traces[:y]) .< 1), digits=3))
println(rpad("exact", 22), "std(y) = ", round(sqrt(2/3), digits=3), ", std(z) = 1.0, P(|y| < 1) = 0.75")

Random.seed!(0)
hmc_traces = hmc(uniform_chain, args, observations, 100_000, 10, 0.1; unconstrained=true, ad_backend=:forwarddiff)
print_summary("HMC", hmc_traces)

# In the original space, HMC proposals regularly leave the support and are rejected (log density -Inf):
# the acceptance rate drops from over 99% to about 18%, so the chain mixes slower and the estimates are less accurate.
Random.seed!(0)
traces = hmc(uniform_chain, args, observations, 100_000, 10, 0.1; unconstrained=false, ad_backend=:forwarddiff)
print_summary("HMC, constrained", traces)

# The mean-field Gaussian is fitted in unconstrained space, sample_posterior transforms back.
# It can only approximate the Logistic distributions of the transformed variables, so it is close but not exact.
Random.seed!(0)
vi_result = advi_meanfield(uniform_chain, args, observations, 10_000, 10, 0.01)
advi_traces = sample_posterior(vi_result, 100_000)
print_summary("ADVI mean-field", advi_traces)

Random.seed!(0)
vi_result = bbvi(uniform_chain, args, observations, 10_000, 10, 0.01)
bbvi_traces = sample_posterior(vi_result, 100_000)
print_summary("BBVI", bbvi_traces)

p = histogram(hmc_traces[:y], normalize=true, label="HMC", linewidth=0, alpha=0.5, xlabel="y")
histogram!(p, traces[:y], normalize=true, label="HMC, constrained", linewidth=0, alpha=0.5)
histogram!(p, advi_traces[:y], normalize=true, label="ADVI", linewidth=0, alpha=0.5)
plot!(p, -2:0.01:2, y -> (2 - abs(y)) / 4, label="exact", color=:black)
show_plot(p, "static_constrained_support")
