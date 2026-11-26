# Urn with an unknown number of balls (Milch et al., Approximate Inference for Infinite Contingent Bayesian Networks,
# AISTATS 2005), see common/milch_urn.jl for the setup and the exact posterior.
#
# The number of random variables depends on N, which a universal model can express directly.
# See graph/milch_urn.jl for exact inference with one graph-based model per N.
#
# Run: julia --project=examples examples/universal/milch_urn.jl [noise]

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using StatsPlots
include("../common/milch_urn.jl")
include("../common/plotting.jl")

@ppl function urn(K::Int)
    N ~ Poisson(6)
    balls = Bool[]
    for i in 1:N
        ball = {:ball => i} ~ Bernoulli(0.5)
        push!(balls, ball)
    end
    for k in 1:K
        if N > 0
            ball_ix = {:drawn_ball => k} ~ DiscreteUniform(1, N)
            {:color => k} ~ Bernoulli(balls[ball_ix] ? 1 - noise : noise)
        else
            # No ball can be drawn from an empty urn, so the observed colours have probability zero
            # (observing a Dirac is a hard constraint).
            {:color => k} ~ Dirac(-1.)
        end
    end
    return N
end

# Without noise, only runs that reproduce the exact sequence of colours get a non-zero weight (~1 in 2000),
# so the effective sample size drops from ~200,000 (noise = 0.2) to ~3,000 (noise = 0).
Random.seed!(0)
Ns, logw = likelihood_weighting(urn, (K,), Observations((:color => k) => colors[k] for k in 1:K), 5_000_000; completion=Evaluation.retval_completion)
W = exp.(logw)
lw_posterior = [sum(W[Ns .== N]) for N in 0:N_max]
println("Effective sample size: ", round(Int, 1 / sum(W .^ 2)))

println("P(N | colours of 10 draws, noise = $noise) for N = 0:", N_max)
println("  exact                ", round.(exact_posterior(), digits=3))
println("  likelihood weighting ", round.(lw_posterior, digits=3))

p = bar(0:N_max, lw_posterior, label="likelihood weighting", xlabel="number of balls N", ylabel="P(N | data)", xticks=0:N_max, linewidth=0, alpha=0.6)
scatter!(p, 0:N_max, exact_posterior(), label="exact", color=:black)
show_plot(p, "universal_milch_urn_noise_$noise")
