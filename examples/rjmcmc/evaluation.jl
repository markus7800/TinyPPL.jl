using TinyPPL.Distributions
using TinyPPL.Evaluation
using TinyPPL.Logjoint
import Random

import LinearAlgebra: logabsdet, det, inv

@ppl function model()
    X ~ Normal(0.,1.)
    Y ~ Normal(0.,1)
end
observations = Dict()

@ppl function aux_static_lmh(tr::Dict{Any,Real}, q::Dict{Any,Distribution})
    n = length(tr)
    chosen_ix ~ DiscreteUniform(1,n)
    chosen_addr = collect(keys(tr))[chosen_ix]

    proposal = q[chosen_addr]
    new_value ~ proposal
end

function static_lmh_transformation!(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}, new_proposal_trace::Dict{Any,Real})
    
    copy_at_address(tt, old_proposal_trace, new_proposal_trace, :chosen_ix)
    chosen_ix = read_discrete(tt, old_proposal_trace, :chosen_ix)
    chosen_addr = collect(keys(old_model_trace))[chosen_ix]

    for (addr, _) in old_model_trace
        if addr != chosen_addr
            copy_at_address(tt, old_model_trace, new_model_trace, addr)
        end
    end
    old_value = read_continuous(tt, old_model_trace, chosen_addr)
    new_value = read_continuous(tt, old_proposal_trace, :new_value)
    write_continuous(tt, new_model_trace, chosen_addr, new_value)
    write_continuous(tt, new_proposal_trace, :new_value, old_value)
end

transformation = TraceTransformation(static_lmh_transformation!)

sampler = Evaluation.TraceSampler()
model((), sampler, observations)
old_model_trace = sampler.X
old_p = sampler.W

sampler = Evaluation.TraceSampler()
q = Dict{Any,Distribution}(:X=>Normal(),:Y=>Normal())
aux_static_lmh((old_model_trace, q), sampler, Dict())
old_proposal_trace = sampler.X
old_q = sampler.W

new_model_trace, new_proposal_trace = apply(transformation, old_model_trace, old_proposal_trace)
J = jacobian(transformation, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace) # at old_trace

sampler = Evaluation.TraceSampler(X = new_model_trace)
model((), sampler, observations)
new_p = sampler.W

sampler = Evaluation.TraceSampler(X = new_proposal_trace)
aux_static_lmh((new_model_trace, q), sampler, Dict())
new_q = sampler.W

new_p + new_q - old_p - new_q + logabsdet(J)[1]


import ForwardDiff
function get_grad_U_fwd_diff(logjoint::Function)
    function grad_U(X::AbstractVector{<:Real})
        grad = ForwardDiff.gradient(logjoint, X)
        return -grad # U = -logjoint
    end
    return grad_U
end

@ppl function aux_static_hmc(tr::Dict{Any,Real})
    for i in 1:length(tr)
        {(:P, i)} ~ Normal(0.,1)
    end
end

ulj = Evaluation.make_unconstrained_logjoint(model, (), Dict())
ulj(old_model_trace)

grad_U = get_grad_U_fwd_diff(X -> ulj(Dict{Any,Real}(:X=>X[1], :Y=>X[2])))
grad_U([old_model_trace[:X], old_model_trace[:Y]])


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

function hmc_transformation!(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}, new_proposal_trace::Dict{Any,Real})

    P = [read_continuous(tt, old_proposal_trace, (:P, i)) for i in 1:length(old_model_trace)]
    X = [read_continuous(tt, old_model_trace, addr) for addr in keys(old_model_trace)]

    X_new, P_new = leapfrog(grad_U, X, P, 10, 0.1)

    for i in 1:length(old_model_trace)
        write_continuous(tt, new_proposal_trace, (:P, i), P_new[i])
    end
    for (i, addr) in enumerate(keys(old_model_trace))
        write_continuous(tt, new_model_trace, addr, X_new[i])
    end
end



transformation = TraceTransformation(hmc_transformation!)

sampler = Evaluation.TraceSampler()
model((), sampler, observations)
old_model_trace = sampler.X
old_p = sampler.W

sampler = Evaluation.TraceSampler()
aux_static_hmc((old_model_trace,), sampler, Dict())
old_proposal_trace = sampler.X
old_q = sampler.W

new_model_trace, new_proposal_trace = apply(transformation, old_model_trace, old_proposal_trace)
J = Evaluation.jacobian_fwd_diff(transformation, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
det(J)
old_model_trace_2, old_proposal_trace_2 = apply(transformation, new_model_trace, new_proposal_trace)
all(old_model_trace[addr] ≈ old_model_trace_2[addr] for addr in keys(old_model_trace)) && all(old_proposal_trace[addr] ≈ old_proposal_trace[addr] for addr in keys(old_proposal_trace))

sampler = Evaluation.TraceSampler(X = new_model_trace)
model((), sampler, observations)
new_p = sampler.W

sampler = Evaluation.TraceSampler(X = new_proposal_trace)
aux_static_lmh((new_model_trace, q), sampler, Dict())
new_q = sampler.W

new_p + new_q
old_p + new_q

new_p + new_q - old_p - new_q + logabsdet(J)[1]




function box_muller!(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real})
    u1 = read_continuous(tt, old_trace, :u1)
    u2 = read_continuous(tt, old_trace, :u2)
    r = sqrt(-2*log(u1))
    theta = 2*pi*u2
    write_continuous(tt, new_trace, :x, r * cos(theta))
    write_continuous(tt, new_trace, :y, r * sin(theta))
end

function box_muller_inv!(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real})
    x = read_continuous(tt, old_trace, :x)
    y = read_continuous(tt, old_trace, :y)
    r2 = x^2 + y^2
    theta = atan(y, x)
    write_continuous(tt, new_trace, :u1, exp(-r2/2))
    write_continuous(tt, new_trace, :u2, theta / (2*pi))
end


transformation = TraceTransformation(box_muller!)
inv_transformation = TraceTransformation(box_muller_inv!)

old_trace = Dict{Any,Real}(:u1 => 0.5, :u2 => 0.3)
new_trace = apply(transformation, old_trace)
J = jacobian(transformation, old_trace, new_trace)
old_trace_2 = apply(inv_transformation, new_trace)
J_inv = jacobian(inv_transformation, new_trace, old_trace_2)
all(inv(J_inv) .≈ J)

logpdf(Normal(0,1),new_trace[:x]) + logpdf(Normal(0,1),new_trace[:y])
logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2]) + logabsdet(J_inv)[1]
logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2]) - logabsdet(J)[1]


begin
    old_trace = Dict{Any,Real}(:u1 => rand(), :u2 => rand())
    new_trace = apply(transformation, old_trace)
    J = jacobian(transformation, old_trace, new_trace)
    lp1 = logpdf(Normal(0,1),new_trace[:x]) + logpdf(Normal(0,1),new_trace[:y])
    lp2 = logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2]) - logabsdet(J)[1]
    lp1 ≈ lp2
end