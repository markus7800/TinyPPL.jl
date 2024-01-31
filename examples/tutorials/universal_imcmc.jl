using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using Plots

@ppl function model()
    X ~ Normal(0.,1.)
    Y ~ Normal(X^2 - 2, 1)
end
observations = Observations()

# We can implement MH, in particular LMH, as iMCMC
# this only works for static models as we would need to provide samples for new addresses in aux model otherwise.

@ppl function static_lmh_aux(tr::UniversalTrace, addr2proposal::Addr2Proposal)
    n = length(tr)
    chosen_ix ~ DiscreteUniform(1,n)
    chosen_addr = sort(collect(keys(tr)))[chosen_ix]

    x_current = tr[chosen_addr]
    dist = proposal_dist(addr2proposal[chosen_addr], x_current)
    proposed_value ~ dist
end

function static_lmh_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)
    
    copy_at_address(tt, old_proposal_trace, new_proposal_trace, :chosen_ix)
    chosen_ix = read_discrete(tt, old_proposal_trace, :chosen_ix)
    chosen_addr = sort(collect(keys(old_model_trace)))[chosen_ix]

    # copy all values for addresses != chosen_addr
    for (addr, _) in old_model_trace
        if addr != chosen_addr
            copy_at_address(tt, old_model_trace, new_model_trace, addr)
        end
    end

    current_value = read_continuous(tt, old_model_trace, chosen_addr)
    proposed_value = read_continuous(tt, old_proposal_trace, :proposed_value)

    # write proposed value to new model trace
    write_continuous(tt, new_model_trace, chosen_addr, proposed_value)
    # write old value to backward proposal trace to reverse LMH update
    write_continuous(tt, new_proposal_trace, :proposed_value, current_value)
end

Q = Addr2Proposal(:X => StaticProposal(Normal()), :Y => ContinuousRandomWalkProposal(0.5, -Inf, Inf))

Random.seed!(0)
traces, lps = imcmc(
    model, (), observations,
    static_lmh_aux, (Q,),
    static_lmh_involution!,
    100_000;
    check_involution=true
);


histogram(traces[:X], normalize=true, lw=0, alpha=0.5, legend=false);
plot!(x -> exp(logpdf(Normal(),x)))

histogram2d(traces[:X], traces[:Y], normalize=true, aspect_ratio=:equal, xlims=(-5,5), ylims=(-5,5))

xs = -5:0.1:5
ys = -5:0.1:5
joint = exp.([score_trace(model, (), observations, UniversalTrace(:X=>x,:Y=>y)) for y in ys, x in xs]);
contour(xs, ys, joint, aspect_ratio=:equal, xlims=(-5,5), ylims=(-5,5))

# We can also implement HMC

import ForwardDiff

function get_grad_U_fwd_diff(logjoint::Function)
    function grad_U(X::AbstractVector{<:Real})
        grad = ForwardDiff.gradient(logjoint, X)
        return -grad # U = -logjoint
    end
    return grad_U
end

function leapfrog(
    grad_U::Function,
    X::AbstractVector{<:Real}, P::AbstractVector{<:Real},
    L::Int, eps::Float64
    )

    P = P - eps/2 * grad_U(X)
    for _ in 1:(L-1)
        X = X + eps * P
        P = P - eps * grad_U(X)
    end
    X = X + eps * P
    P = P - eps/2 * grad_U(X)

    return X, -P
end

@ppl function hmc_aux(tr::UniversalTrace)
    # sample momentum
    for i in 1:length(tr)
        {(:P, i)} ~ Normal(0.,1)
    end
end


# maps UniversalTrace -> Float64
logjoint = Evaluation.make_unconstrained_logjoint(model, (), observations)
# grad needs Array -> Float64
grad_U = get_grad_U_fwd_diff(arr -> logjoint(UniversalTrace(:X=>arr[1], :Y=>arr[2])))


function hmc_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    # read momentum
    P = [read_continuous(tt, old_proposal_trace, (:P, i)) for i in 1:length(old_model_trace)]
    # read position
    X = [read_continuous(tt, old_model_trace, addr) for addr in keys(old_model_trace)]

    # leapfrog to new position, P_new is backwards momentum
    X_new, P_new = leapfrog(grad_U, X, P, 10, 0.1)

    # write backwards momentum
    for i in 1:length(old_model_trace)
        write_continuous(tt, new_proposal_trace, (:P, i), P_new[i])
    end

    # write new position
    for (i, addr) in enumerate(keys(old_model_trace))
        # if model has constraint parameters, we would need to transform here
        write_continuous(tt, new_model_trace, addr, X_new[i])
    end
end

import LinearAlgebra: det
hmc_transformation = TraceTransformation(hmc_involution!)

old_model_trace = sample_trace(model, (), observations)
old_model_score = score_trace(model, (), observations, old_model_trace)

old_proposal_trace = sample_trace(hmc_aux, (old_model_trace,))
old_proposal_score = score_trace(hmc_aux, (old_model_trace,), old_proposal_trace)

new_model_trace, new_proposal_trace = apply(hmc_transformation, old_model_trace, old_proposal_trace)
J = jacobian(hmc_transformation, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
# leapfrog makes det = 1
det(J)
# leapfrog is reversible
old_model_trace_2, old_proposal_trace_2 = apply(hmc_transformation, new_model_trace, new_proposal_trace)
@assert all(old_model_trace[addr] ≈ old_model_trace_2[addr] for addr in keys(old_model_trace)) && all(old_proposal_trace[addr] ≈ old_proposal_trace[addr] for addr in keys(old_proposal_trace))



Random.seed!(0)
traces, lps = imcmc(
    model, (), observations,
    hmc_aux, (),
    hmc_involution!,
    100_000;
    check_involution=true
);


histogram(traces[:X], normalize=true, lw=0, alpha=0.5, legend=false);
plot!(x -> exp(logpdf(Normal(),x)))

histogram2d(traces[:X], traces[:Y], normalize=true, aspect_ratio=:equal, xlims=(-5,5), ylims=(-5,5))

contour(xs, ys, joint, aspect_ratio=:equal, xlims=(-5,5), ylims=(-5,5))
