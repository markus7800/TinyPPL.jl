# SMC and particle Gibbs on a linear Gaussian state space model, as static model
# (see common/lgss.jl for the model and the exact posterior).
#
# For evaluation-based models, SMC pauses the program at each observation and copies its execution state with Libtask.
# This is much slower than for graph-based models, which traverse the nodes directly,
# so we only use the first T = 50 time steps here. See graph/lgss.jl for T = 400 and a comparison of the algorithms.
#
# Run: julia --project=examples examples/static/lgss.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("../common/lgss.jl")

T = 50
posterior_μ, posterior_std, log_evidence = lgss_exact(y[1:T])

@ppl static function lgss(T)
    x = 0.
    for t in 1:T
        x = {:x => t} ~ Normal(a * x, σ_v)
        {:y => t} ~ Normal(x, σ_e)
    end
end

args = (T,)
observations = Observations((:y => t) => y[t] for t in 1:T)
# T × n_samples matrix of the states
states(traces) = reduce(vcat, [traces[:x => t]' for t in 1:T])

Random.seed!(0)
traces, logw, log_Z = smc(lgss, args, observations, 1_000)
println("SMC: log p(y) ≈ ", round(log_Z, digits=2), " (exact: ", round(log_evidence, digits=2), ")")

# Particle Gibbs with ancestral sampling, see graph/lgss.jl.
Random.seed!(0)
traces = particle_gibbs(lgss, args, observations, 5, 500; ancestral_sampling=true)
X = states(traces)
mean_error = maximum(abs, vec(mean(X[:, 51:end], dims=2)) .- posterior_μ) # 50 samples burn-in
println("particle Gibbs, N = 5, ancestral sampling: mean update rate ", round(mean(update_rate(X)), digits=2),
    ", max. error of posterior mean: ", round(mean_error, digits=3))

# A custom proposal is a ProposalDistribution whose proposal_dist method returns the distribution for an address,
# given the current trace X and the map from addresses to indices into X.
# Here, we propose from the locally optimal proposal p(x_t | x_{t-1}, y_t).
struct LGSSProposal <: Distributions.ProposalDistribution end

function Distributions.proposal_dist(::LGSSProposal, x_current::Tuple{Pair,StaticTrace,Evaluation.Addr2Ix})
    addr, X, addresses_to_ix = x_current
    t::Int = addr[2]
    x_last = t == 1 ? 0. : X[addresses_to_ix[:x => (t-1)]]
    return optimal_proposal(x_last, y[t])
end

addr2proposal = Addr2Proposal((:x => t) => LGSSProposal() for t in 1:T)

Random.seed!(0)
traces, logw, log_Z = smc(lgss, args, observations, 1_000; addr2proposal=addr2proposal)
println("SMC with optimal proposal: log p(y) ≈ ", round(log_Z, digits=2))

Random.seed!(0)
traces = particle_gibbs(lgss, args, observations, 5, 500; ancestral_sampling=true, addr2proposal=addr2proposal)
X = states(traces)
mean_error = maximum(abs, vec(mean(X[:, 51:end], dims=2)) .- posterior_μ)
println("particle Gibbs with optimal proposal: mean update rate ", round(mean(update_rate(X)), digits=2),
    ", max. error of posterior mean: ", round(mean_error, digits=3))
