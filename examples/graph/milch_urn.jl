# Urn with an unknown number of balls (Milch et al., Approximate Inference for Infinite Contingent Bayesian Networks,
# AISTATS 2005), see common/milch_urn.jl for the setup and the exact posterior.
#
# A graph-based model has a fixed set of variables, so it cannot have a random number of balls.
# Instead, we compile one model per N, compute its exact evidence p(data | N) with variable elimination on the factor graph,
# and combine it with the prior p(N). See universal/milch_urn.jl for the same model with a random N.
#
# Run: julia --project=examples examples/graph/milch_urn.jl [noise]

using TinyPPL.Distributions
using TinyPPL.Graph
include("../common/milch_urn.jl")

# Each observation only depends on the colours of the balls and on one drawn index, so its factor has 2^N * N entries.
# Observing the number of black balls instead (noise-free), `Dirac(sum(...)) ↦ 5`, would create a single factor over all
# balls and drawn indices with 2^N * N^K entries, which is infeasible already for N = 6 (~4 * 10^9 entries).
function urn_model(N::Int)
    @pgm urn begin
        let balls = [{:ball => i} ~ Bernoulli(0.5) for i in 1:$(N)],
            drawn = [{:drawn_ball => k} ~ DiscreteUniform(1, $(N)) for k in 1:$(K)]

            [{:color => k} ~ Bernoulli(balls[Int(drawn[k])] == 1. ? $(1 - noise) : $(noise)) ↦ $(colors)[k] for k in 1:$(K)]
            0
        end
    end
end

# The cost of variable elimination grows exponentially with N (~2s for N = 10, ~25s for N = 13),
# so we stop at N = 12 and compare to the exact posterior truncated at the same N.
N_max_graph = 12
println("log p(data | N), exact vs. variable elimination:")
log_joint = [-Inf] # N = 0 is impossible, as no ball can be drawn
for N in 1:N_max_graph
    log_evidence = greedy_variable_elimination(urn_model(N)).log_evidence
    println("  N = ", lpad(N, 2), ": ", round(log_likelihood(N), digits=4), " vs. ", round(log_evidence, digits=4))
    push!(log_joint, logpdf(Poisson(6), N) + log_evidence)
end
posterior = exp.(log_joint .- logsumexp(log_joint))

println("P(N | data, N ≤ $N_max_graph, noise = $noise) for N = 0:", N_max_graph)
println("  exact                ", round.(exact_posterior(N_max_graph), digits=3))
println("  variable elimination ", round.(posterior, digits=3))
