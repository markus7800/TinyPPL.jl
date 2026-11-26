# Sequential Monte Carlo and particle Gibbs on a linear Gaussian state space model with T = 400 time steps
# (see common/lgss.jl for the model and the exact posterior).
#
# - Likelihood weighting collapses on long sequences, SMC estimates the evidence well by resampling after each observation.
# - Particle Gibbs (Andrieu, Doucet, Holenstein: Particle Markov chain Monte Carlo methods, JRSS-B 2010) uses
#   conditional SMC, which keeps a reference trajectory, as MCMC kernel. Because of path degeneracy, it rarely updates
#   early states. Ancestral sampling (Lindsten, Jordan, Schön: Particle Gibbs with Ancestor Sampling, JMLR 2014)
#   resamples the reference trajectory's ancestors and mixes well even with 5 particles.
# - Custom proposals for SMC and particle Gibbs.
# See static/lgss.jl for the same algorithms on an evaluation-based model.
#
# Run: julia --project=examples examples/graph/lgss.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import Random
using Plots
include("../common/lgss.jl")
include("../common/plotting.jl")

T = 400
posterior_μ, posterior_std, log_evidence = lgss_exact(y[1:T])

model = @pgm lgss begin
    function step(t, x, a, σ_v, σ_e, y)
        let new_x = {:x => t} ~ Normal(a * x, σ_v)
            {:y => t} ~ Normal(new_x, σ_e) ↦ y[t]
            new_x
        end
    end
    let y = $(y)
        @loop($(T), step, 0., $(a), $(σ_v), $(σ_e), y)
    end
end
# The latent node with index t is x_t, so traces.data has one row per time step.
@assert all(get_address(model, t) == (:x => t) for t in 1:T)


# ===== Likelihood weighting vs. SMC =====
println("log p(y): exact = ", round(log_evidence, digits=2))

# The weights are products of 400 likelihoods, so a single sample gets almost all the weight.
Random.seed!(0)
traces, logw, log_Z = likelihood_weighting(model, 10_000)
println("likelihood weighting ≈ ", round(log_Z, digits=2), " (effective sample size: ", round(1 / sum(exp.(logw) .^ 2), digits=1), ")")

# light_smc is a minimal implementation of smc that always resamples.
Random.seed!(0)
traces, logw, log_Z = light_smc(model, 10_000)
println("light SMC ≈ ", round(log_Z, digits=2))

# Resampling makes the particles share their early history (path degeneracy),
# so SMC estimates the evidence and the last states well, but not the earlier states.
Random.seed!(0)
traces, logw, log_Z = smc(model, 10_000)
smc_mean = traces.data * exp.(logw)
println("SMC ≈ ", round(log_Z, digits=2), ", max. error of posterior mean: ", round(maximum(abs, smc_mean .- posterior_μ), digits=3),
    " (", length(unique(traces.data[1, :])), " distinct values of x_1 among 10000 particles)")


# ===== Particle Gibbs with and without ancestral sampling =====
# Each iteration runs conditional SMC, which keeps the current trajectory as one of the particles, and samples
# a new trajectory from the final particles. With N particles, a state is updated with probability at most (N-1)/N.
# Without ancestral sampling, the early states stay at their initial value 0, which gives the same large error for all N.
#
# Even the last state x_T is updated less often than (N-1)/N. The new trajectory is sampled proportionally to the final
# weights, and each particle would only be chosen with probability 1/N if all weights were equal, but they are not.
# With proposal r, the weight of a particle at time t is
#   W(x_1:t) = p(x_1:t, y_1:t) / (p(x_1:t-1, y_1:t-1) r(x_t | x_1:t-1))
#            = p(x_t, y_t | x_1:t-1, y_1:t-1) / r(x_t | x_1:t-1)
#            = p(x_t, y_t | x_t-1) / r(x_t | x_1:t-1)          (Markov property)
# which depends on the particle. Even for the locally optimal proposal r(x_t | x_1:t-1) = p(x_t | x_t-1, y_t)
# (see below), W(x_1:t) = p(y_t | x_t-1) still depends on x_t-1.
n_samples = 1_000
p = plot(xlabel="t", ylabel="update rate", ylims=(0, 1), legend=:left)
for (i, ancestral_sampling) in enumerate([false, true]), N in [5, 20]
    Random.seed!(0)
    local traces = particle_gibbs(model, N, n_samples; ancestral_sampling=ancestral_sampling, init=:zeros)
    rates = update_rate(traces.data)
    local mean_error = maximum(abs, vec(mean(traces.data[:, 101:end], dims=2)) .- posterior_μ) # 100 samples burn-in
    println("particle Gibbs, N = ", lpad(N, 2), ", ancestral sampling = ", rpad(ancestral_sampling, 5),
        ": update rate of x_1 / x_200 / x_400 = ", round.(rates[[1, 200, 400]], digits=2), ", max. error of posterior mean: ", round(mean_error, digits=3))
    plot!(p, rates, label="N = $N" * (ancestral_sampling ? ", ancestral sampling" : ""), linestyle=ancestral_sampling ? :solid : :dash)
    ancestral_sampling && N == 5 && global pgas_traces = traces
end
show_plot(p, "graph_lgss_update_rates")

μ = vec(mean(pgas_traces.data[:, 101:end], dims=2))
σ = vec(std(pgas_traces.data[:, 101:end], dims=2))
p = plot(posterior_μ, ribbon=2posterior_std, label="exact posterior (±2 std)", fillalpha=0.3, xlabel="t")
plot!(p, μ, ribbon=2σ, label="particle Gibbs with ancestral sampling, N = 5", fillalpha=0.3)
plot!(p, x_gt, label="ground truth", color=:black, linewidth=0.5)
show_plot(p, "graph_lgss_posterior")


# ===== Custom proposals =====
# By default, SMC proposes each state from its prior p(x_t | x_{t-1}). A proposal is a ProposalDistribution
# whose proposal_dist method returns the distribution for a node, given the current values of all latent nodes.
# Here, we propose from the locally optimal proposal p(x_t | x_{t-1}, y_t), which takes the observation into account.
struct LGSSProposal <: Distributions.ProposalDistribution end

function Distributions.proposal_dist(::LGSSProposal, x_current::Tuple{Pair,Vector{Float64}})
    addr, X = x_current
    t::Int = addr[2]
    x_last = t == 1 ? 0. : X[t-1]
    return optimal_proposal(x_last, y[t])
end

addr2proposal = Addr2Proposal((:x => t) => LGSSProposal() for t in 1:T)

Random.seed!(0)
traces, logw, log_Z = smc(model, 10_000; addr2proposal=addr2proposal)
println("\nSMC with optimal proposal: log p(y) ≈ ", round(log_Z, digits=2), " (exact: ", round(log_evidence, digits=2), ")")

Random.seed!(0)
traces = particle_gibbs(model, 5, n_samples; ancestral_sampling=true, init=:zeros, addr2proposal=addr2proposal)
mean_error = maximum(abs, vec(mean(traces.data[:, 101:end], dims=2)) .- posterior_μ)
println("particle Gibbs with optimal proposal, N = 5, ancestral sampling: max. error of posterior mean: ", round(mean_error, digits=3))


# ===== Direct implementation of particle Gibbs with ancestral sampling =====
# Particle Gibbs written directly for the state space model, without the PPL (Lindsten et al., Algorithm 2).
# With the same seed, it gives exactly the same update rates and errors as particle_gibbs above, which checks the PPL implementation.
import TinyPPL: logsumexp

transition(x_last) = Normal(a * x_last, σ_v) # x_0 = 0
emission(x) = Normal(x, σ_e)
proposal(x_last, y_t, propose_from_posterior) = propose_from_posterior ? optimal_proposal(x_last, y_t) : transition(x_last)
normalised(log_w) = exp.(log_w .- logsumexp(log_w))

# One conditional SMC sweep with reference trajectory x_ref (always particle 1), returns the new trajectory.
function pgas_kernel(x_ref, y, N; ancestral_sampling, propose_from_posterior)
    T = length(y)
    particles = Matrix{Float64}(undef, T, N)
    log_w = zeros(N)
    for t in 1:T
        if t > 1
            A = rand(Categorical(normalised(log_w)), N)
            if ancestral_sampling
                # resample the ancestor of the reference proportionally to w_i p(x_ref[t] | x_i[t-1])
                A[1] = rand(Categorical(normalised(log_w .+ [logpdf(transition(particles[t-1, i]), x_ref[t]) for i in 1:N])))
            else
                A[1] = 1
            end
            particles = particles[:, A]
        end
        for i in 1:N
            x_last = t == 1 ? 0. : particles[t-1, i]
            q = proposal(x_last, y[t], propose_from_posterior)
            x = i == 1 ? x_ref[t] : rand(q)
            particles[t, i] = x
            log_w[i] = logpdf(emission(x), y[t]) + logpdf(transition(x_last), x) - logpdf(q, x)
        end
    end
    k = rand(Categorical(normalised(log_w)))
    return particles[:, k]
end

function pgas(y, N, n_samples; ancestral_sampling=true, propose_from_posterior=false)
    X = zeros(length(y), n_samples)
    x_current = zeros(length(y))
    for i in 1:n_samples
        x_current = pgas_kernel(x_current, y, N; ancestral_sampling=ancestral_sampling, propose_from_posterior=propose_from_posterior)
        X[:, i] = x_current
    end
    return X
end

println()
for (ancestral_sampling, propose_from_posterior) in [(false, false), (true, false), (true, true)]
    Random.seed!(0)
    X = pgas(y[1:T], 5, n_samples; ancestral_sampling=ancestral_sampling, propose_from_posterior=propose_from_posterior)
    rates = update_rate(X)
    local mean_error = maximum(abs, vec(mean(X[:, 101:end], dims=2)) .- posterior_μ)
    println("direct PGAS, N = 5, ancestral sampling = ", rpad(ancestral_sampling, 5), ", optimal proposal = ", rpad(propose_from_posterior, 5),
        ": update rate of x_1 / x_200 / x_400 = ", round.(rates[[1, 200, 400]], digits=2), ", max. error of posterior mean: ", round(mean_error, digits=3))
end
