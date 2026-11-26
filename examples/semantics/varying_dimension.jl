# A variable that only exists in one branch, and what goes wrong if such a model is declared static.
#
# x ~ Uniform(-1, 1), and only if x <= 0 we sample y ~ Uniform(0, size). Nothing is observed, so x is uniform
# and P(x <= 0) = 0.5. Traces for x > 0 have one variable and traces for x <= 0 have two.
# Universal models handle this, and their MCMC algorithms are correct.
#
# A static model promises that every run has the same addresses (this is not checked).
# Declaring this model static anyway makes y part of every trace. For x > 0, the model does not score y,
# so y has no density at all there (equivalently, a flat density on ℝ), and the target over (x, y) cannot be normalised.
# HMC then gives wrong results, which even depend on the sampler settings.
# See universal/discontinuous_vi.jl for a different issue with branching: discontinuous densities.
#
# Run: julia --project=examples examples/semantics/varying_dimension.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots
include("../common/plotting.jl")

@ppl function gap_model(size)
    x ~ Uniform(-1., 1.)
    if x <= 0
        y ~ Uniform(0., size)
    end
end

args = (2.,)
observations = Observations()

# The log joint of a trace with one variable is a density w.r.t. a 1-dimensional measure, with two variables
# w.r.t. a 2-dimensional measure, so the values 0.5 and 0.25 cannot be compared directly.
logjoint = Evaluation.make_logjoint(gap_model, args, observations)
println("p(x = 0.5) = ", exp(logjoint(UniversalTrace(:x => 0.5))), ", p(x = -0.5, y = 1) = ", exp(logjoint(UniversalTrace(:x => -0.5, :y => 1.))))

Random.seed!(0)
lmh_traces = lmh(gap_model, args, observations, 1_000_000)
println("universal, LMH:  P(x <= 0) ≈ ", round(mean(lmh_traces[:x] .<= 0), digits=3), " (exact: 0.5)")

Random.seed!(0)
rwmh_traces = rwmh(gap_model, args, observations, 1_000_000)
println("universal, RWMH: P(x <= 0) ≈ ", round(mean(rwmh_traces[:x] .<= 0), digits=3), " (exact: 0.5)")

# The same model, wrongly declared static.
@ppl static function static_gap_model(size)
    x ~ Uniform(-1., 1.)
    if x <= 0
        y ~ Uniform(0., size)
    end
end

# hmc(model, args, observations, n_samples, L, eps): with more leapfrog steps L, y drifts further and P(x <= 0) changes.
for L in [1, 10]
    Random.seed!(0)
    traces = hmc(static_gap_model, args, observations, 200_000, L, 0.01, ad_backend=:forwarddiff)
    println("static, HMC with L = ", lpad(L, 2), ": P(x <= 0) ≈ ", round(mean(traces[:x] .<= 0), digits=3),
        ", y ranges over ", round.(extrema(traces[:y]), digits=1))
    L == 10 && global hmc_traces = traces
end

p = histogram(rwmh_traces[:x], normalize=true, label="universal, RWMH", xlabel="x", linewidth=0, alpha=0.5)
histogram!(p, hmc_traces[:x], normalize=true, label="static, HMC", linewidth=0, alpha=0.5)
show_plot(p, "semantics_varying_dimension")
