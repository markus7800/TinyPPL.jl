# Involutive MCMC: MH and HMC as special cases
# (Neklyudov, Welling, Egorov, Vetrov: Involutive MCMC: a Unifying Framework, ICML 2020;
# Cusumano-Towner, Lew, Mansinghka: Automating Involutive MCMC using Probabilistic and Differentiable Programming, 2020).
#
# An iMCMC step samples auxiliary variables v ~ q(· | x) from an auxiliary program, applies an involution
# (x', v') = f(x, v), i.e. f(f(x, v)) = (x, v), and accepts with probability
#   min(1, p(x') q(v' | x') / (p(x) q(v | x)) * |det J_f(x, v)|).
# imcmc(model, args, observations, aux_model, aux_args, involution!, n_samples) runs this with an involution written
# as trace transformation (see trace_transform.jl). With check_involution=true, it verifies f(f(x, v)) = (x, v).
#
# Run: julia --project=examples examples/universal/imcmc.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import ForwardDiff
import LinearAlgebra: det
using Plots
include("../common/plotting.jl")

# A banana-shaped distribution: X ~ Normal(0, 1) and Y ~ Normal(X^2 - 2, 1). Nothing is observed, so
# E[X] = 0, std(X) = 1, E[Y] = E[X^2] - 2 = -1, and Var(Y) = Var(X^2) + 1 = 3.
@ppl function banana()
    X ~ Normal(0., 1.)
    Y ~ Normal(X^2 - 2, 1.)
end
observations = Observations()

_std(v) = sqrt(mean((v .- mean(v)) .^ 2))
print_summary(method, traces) = println(rpad(method, 18), "E[X] ≈ ", round(mean(traces[:X]), digits=3), ", std(X) ≈ ", round(_std(traces[:X]), digits=3),
    ", E[Y] ≈ ", round(mean(traces[:Y]), digits=3), ", std(Y) ≈ ", round(_std(traces[:Y]), digits=3))
println(rpad("exact", 18), "E[X] = 0, std(X) = 1, E[Y] = -1, std(Y) = ", round(sqrt(3), digits=3))


# ===== Single-site Metropolis-Hastings =====
# The auxiliary program chooses an address and proposes a new value for it.
# The involution swaps the current and the proposed value, so applying it twice gives the original traces back.
# This only works if the model always has the same addresses,
# otherwise the auxiliary program would also have to sample values for new addresses.
@ppl function lmh_aux(tr::UniversalTrace, addr2proposal::Addr2Proposal)
    chosen_ix ~ DiscreteUniform(1, length(tr))
    chosen_addr = sort(collect(keys(tr)))[chosen_ix]
    proposed_value ~ proposal_dist(addr2proposal[chosen_addr], tr[chosen_addr])
end

function lmh_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    copy_at_address(tt, old_proposal_trace, new_proposal_trace, :chosen_ix)
    chosen_ix = read_discrete(tt, old_proposal_trace, :chosen_ix)
    chosen_addr = sort(collect(keys(old_model_trace)))[chosen_ix]

    for addr in keys(old_model_trace)
        if addr != chosen_addr
            copy_at_address(tt, old_model_trace, new_model_trace, addr)
        end
    end

    current_value = read_continuous(tt, old_model_trace, chosen_addr)
    proposed_value = read_continuous(tt, old_proposal_trace, :proposed_value)
    write_continuous(tt, new_model_trace, chosen_addr, proposed_value)
    write_continuous(tt, new_proposal_trace, :proposed_value, current_value)
end

# Propose X from its prior and Y with a random walk. Updating one variable at a time mixes slowly,
# because X and Y are strongly dependent, so we need more samples than for HMC below.
addr2proposal = Addr2Proposal(:X => StaticProposal(Normal()), :Y => ContinuousRandomWalkProposal(2., -Inf, Inf))

Random.seed!(0)
lmh_traces = imcmc(banana, (), observations, lmh_aux, (addr2proposal,), lmh_involution!, 300_000; check_involution=true)
print_summary("iMCMC as LMH", lmh_traces)


# ===== Hamiltonian Monte Carlo =====
# The auxiliary program samples a momentum P ~ Normal(0, I). The involution runs L leapfrog steps and negates
# the momentum. Leapfrog is volume preserving (det J = 1) and reversible, so with the negated momentum,
# running it again returns to the start.
@ppl function hmc_aux(tr::UniversalTrace)
    for i in 1:length(tr)
        {:P => i} ~ Normal(0., 1.)
    end
end

# Potential energy U = -log p, as a function of the vector of values in the order of keys(trace).
logjoint = Evaluation.make_unconstrained_logjoint(banana, (), observations)
addresses = [:X, :Y]
grad_U(X) = -ForwardDiff.gradient(x -> logjoint(UniversalTrace(addr => x[i] for (i, addr) in enumerate(addresses))), X)

function leapfrog(X, P, L, eps)
    P = P - eps / 2 * grad_U(X)
    for _ in 1:(L-1)
        X = X + eps * P
        P = P - eps * grad_U(X)
    end
    X = X + eps * P
    P = P - eps / 2 * grad_U(X)
    return X, -P
end

function hmc_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    # read position and momentum
    X = [read_continuous(tt, old_model_trace, addr) for addr in addresses]
    P = [read_continuous(tt, old_proposal_trace, :P => i) for i in eachindex(addresses)]
    # run leapfrog
    X_new, P_new = leapfrog(X, P, 10, 0.1)
    # write back new position and momentum
    # the model has no constrained variables, otherwise we would have to transform X_new
    for (i, addr) in enumerate(addresses)
        write_continuous(tt, new_model_trace, addr, X_new[i])
        write_continuous(tt, new_proposal_trace, :P => i, P_new[i])
    end
end

Random.seed!(0)
hmc_transformation = TraceTransformation(hmc_involution!)
model_trace = sample_trace(banana, (), observations)
aux_trace = sample_trace(hmc_aux, (model_trace,))
new_model_trace, new_aux_trace = apply(hmc_transformation, model_trace, aux_trace)
J = jacobian(hmc_transformation, model_trace, aux_trace, new_model_trace, new_aux_trace)
model_trace_2, aux_trace_2 = apply(hmc_transformation, new_model_trace, new_aux_trace)
println("\nleapfrog: det J = ", det(J), ", is an involution: ",
    all(model_trace[addr] ≈ model_trace_2[addr] for addr in keys(model_trace)) &&
    all(aux_trace[addr] ≈ aux_trace_2[addr] for addr in keys(aux_trace)))

Random.seed!(0)
hmc_traces = imcmc(banana, (), observations, hmc_aux, (), hmc_involution!, 100_000; check_involution=true)
print_summary("iMCMC as HMC", hmc_traces)

xs = -3:0.05:3
ys = -5:0.05:5
density = [exp(score_trace(banana, (), observations, UniversalTrace(:X => x, :Y => y))) for y in ys, x in xs]
p = scatter(hmc_traces[:X][1:10:end], hmc_traces[:Y][1:10:end], markersize=1, markerstrokewidth=0, alpha=0.3, label="HMC samples", xlabel="X", ylabel="Y")
contour!(p, xs, ys, density, color=:black, colorbar=false)
show_plot(p, "universal_imcmc_hmc")
