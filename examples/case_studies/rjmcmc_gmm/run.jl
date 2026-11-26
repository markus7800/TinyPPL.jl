# Gaussian mixture model with an unknown number of components, inferred with reversible jump MCMC
# (Richardson, Green: On Bayesian Analysis of Mixtures with an Unknown Number of Components, JRSS-B 1997),
# written as involutive MCMC (see universal/imcmc.jl).
#
# - model.jl: k ~ 1 + Poisson(2) components with stick-breaking weights, means and variances, and an allocation
#   of each of the 100 data points (common/gmm_data.jl, generated with 4 components).
# - aux_model.jl: chooses one of six moves and samples its auxiliary variables: Gibbs updates of the means, variances,
#   allocations and weights, and splitting one component into two or merging two components into one.
# - involution.jl: maps (trace, auxiliary variables) to new ones for each move; split and merge are each other's inverse.
#   The Jacobian of the dimension-changing moves is computed automatically by the trace transformation.
# - helpers.jl: weights ↔ stick-breaking parameters and the split / merge parameter transformations.
#
# Run: julia --project=examples examples/case_studies/rjmcmc_gmm/run.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

include("../../common/gmm_data.jl")
include("../../common/gmm_plotting.jl")
include("model.jl")
include("helpers.jl")
include("aux_model.jl")
include("involution.jl")

# With check_involution=true, imcmc verifies in each step that applying the involution twice gives the original traces.
# This is slow, so we only check a short chain.
Random.seed!(0)
imcmc(gmm, (n,), observations, aux_model, (n,), involution!, 1_000; check_involution=true)
println("involution checked for 1000 steps")

Random.seed!(0)
traces = imcmc(gmm, (n,), observations, aux_model, (n,), involution!, 100_000)
lp = [score_trace(gmm, (n,), observations, X) for X in traces.data] # log p(X, Y) of each sample

println("number of times k changed: ", sum(traces[:k, i] != traces[:k, i+1] for i in 1:(length(traces)-1)))
ks = sort(unique(traces[:k]))
println("P(k | y) ≈ ", join(["P($k) = $(round(mean(traces[:k] .== k), digits=3))" for k in ks], ", "), " (data generated with k = $gt_k)")

# The plotting helper expects the weights at addresses :w => j.
function with_weights(trace)
    trace = copy(trace)
    w = get_w([trace[:phi => j] for j in 1:(trace[:k]-1)])
    for j in 1:trace[:k]
        trace[:w => j] = w[j]
    end
    return trace
end

# The most likely sample for each number of components with posterior probability of at least 1%.
plots = []
for k in ks
    mask = traces[:k] .== k
    mean(mask) < 0.01 && continue
    best = argmax(lp[mask])
    push!(plots, plot(visualize_trace(with_weights(traces.data[mask][best])), title="k = $k, log p = $(round(lp[mask][best], digits=1))"))
end
show_plot(plot(plots..., layout=(length(plots), 1), size=(600, 250 * length(plots))), "rjmcmc_gmm_best_traces")
show_plot(plot(plot(traces[:k], ylabel="k", legend=false), plot(lp, ylabel="log p(X, Y)", ylims=(maximum(lp) - 100, maximum(lp)), legend=false),
    layout=(2, 1), xlabel="iteration", xformatter=:plain), "rjmcmc_gmm_chain")
