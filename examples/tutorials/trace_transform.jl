using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

# =========== Box muller transform (deterministic) =================

function box_muller!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    u1 = read_continuous(tt, old_trace, :u1)
    u2 = read_continuous(tt, old_trace, :u2)
    r = sqrt(-2*log(u1))
    theta = 2*pi*u2
    write_continuous(tt, new_trace, :x, r * cos(theta))
    write_continuous(tt, new_trace, :y, r * sin(theta))
end

function box_muller_inv!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    x = read_continuous(tt, old_trace, :x)
    y = read_continuous(tt, old_trace, :y)
    r2 = x^2 + y^2
    theta = atan(y, x)
    write_continuous(tt, new_trace, :u1, exp(-r2/2))
    write_continuous(tt, new_trace, :u2, theta / (2*pi))
end


transformation = TraceTransformation(box_muller!)
inv_transformation = TraceTransformation(box_muller_inv!)

old_trace = UniversalTrace(:u1 => 0.5, :u2 => 0.3)
new_trace = apply(transformation, old_trace)
J = jacobian(transformation, old_trace, new_trace)

# check inverse
old_trace_2 = apply(inv_transformation, new_trace)
all(old_trace[addr] ≈ old_trace_2[addr] for addr in keys(old_trace))
J_inv = jacobian(inv_transformation, new_trace, old_trace_2)

all(inv(J_inv) .≈ J)

import LinearAlgebra: logabsdet
# p(new_model_trace) = p(old_model_score) - log_abs_det
logpdf(Normal(0,1),new_trace[:x]) + logpdf(Normal(0,1),new_trace[:y]) # we know p(new_model_trace) = p(T(old_model_trace))
logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2]) + logabsdet(J_inv)[1]
logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2]) - logabsdet(J)[1]

# p(old_model_trace) = p(new_model_score) + log_abs_det
logpdf(Normal(0,1),new_trace[:x]) + logpdf(Normal(0,1),new_trace[:y]) + logabsdet(J)[1]
logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2])

begin
    old_trace = UniversalTrace(:u1 => rand(), :u2 => rand())
    new_trace = apply(transformation, old_trace)
    J = jacobian(transformation, old_trace, new_trace)
    lp1 = logpdf(Normal(0,1),new_trace[:x]) + logpdf(Normal(0,1),new_trace[:y])
    lp2 = logpdf(Uniform(0,1),old_trace[:u1]) + logpdf(Uniform(0,1),old_trace[:u2]) - logabsdet(J)[1]
    lp1 ≈ lp2
end



# =========== Non-deterministic transform (proposal) =================



@ppl function p1()
    i ~ DiscreteUniform(1, 10) # interval [(i-1)/10, i/10]
    j ~ DiscreteUniform(1, 10) # interval [(j-1)/10, j/10]
end

@ppl function p2()
    x ~ Normal(0, 0.5)
    y ~ Normal(1, 1)
end

@ppl function q1(p2_trace)
    dx ~ Uniform(0.0, 0.1)
    dy ~ Uniform(0.0, 0.1)
end

@ppl function q2(p1_trace)
end

# space 1 -> space 2
function f!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    i = read_discrete(tt, old_model_trace, :i)
    j = read_discrete(tt, old_model_trace, :j)

    dx = read_continuous(tt, old_proposal_trace, :dx)
    dy = read_continuous(tt, old_proposal_trace, :dy)

    u1 = (i-1)/10 + dx
    u2 = (j-1)/10 + dy

    # box muller
    r = sqrt(-2*log(u1))
    theta = 2*pi*u2
    write_continuous(tt, new_model_trace, :x, r * cos(theta))
    write_continuous(tt, new_model_trace, :y, r * sin(theta))
end

# space 2 -> space 1
function finv!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    x = read_continuous(tt, old_model_trace, :x)
    y = read_continuous(tt, old_model_trace, :y)

    # inv box muller
    r2 = x^2 + y^2
    theta = atan(y, x)
    u1 =  exp(-r2/2)
    u2 = theta / (2*pi)

    i = ceil(u1 * 10)
    j = ceil(u2 * 10)

    write_continuous(tt, new_proposal_trace, :dx, u1 - (i-1)/10)
    write_continuous(tt, new_proposal_trace, :dy, u2 - (j-1)/10)

    write_discrete(tt, new_model_trace, :i , i)
    write_discrete(tt, new_model_trace, :j , j)
end


f_transform = TraceTransformation(f!)
finv_transform = TraceTransformation(finv!)

old_model_trace = sample_trace(p1, ())
old_proposal_trace = sample_trace(q1, (old_model_trace,))

new_model_trace, new_proposal_trace = apply(f_transform, old_model_trace, old_proposal_trace)
J = jacobian(f_transform, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)

# check inverse
old_model_trace_2, old_proposal_trace_2 = apply(finv_transform, new_model_trace, new_proposal_trace)

all(old_model_trace[addr] ≈ old_model_trace_2[addr] for addr in keys(old_model_trace_2))
all(old_proposal_trace[addr] ≈ old_proposal_trace_2[addr] for addr in keys(old_proposal_trace))

# transform again targets Normal(0,1) × Normal(0,1)
old_model_score = score_trace(p1, (), old_model_trace)
old_proposal_score = score_trace(q1, (old_model_trace,), old_proposal_trace)

old_model_score + old_proposal_score - logabsdet(J)[1] # p(T(old_model_trace, old_proposal_trace))
logpdf(Normal(0,1), new_model_trace[:x]) + logpdf(Normal(0,1), new_model_trace[:y]) # again we know this distribution from box muller

# this is different
new_model_score = score_trace(p2, (), new_model_trace)
new_proposal_score = score_trace(q2, (new_model_trace,), new_proposal_trace)

# thus T can be used to construct proposals in the correct support, but importance ratio has to be computed
traces, lps = begin
    Random.seed!(0)
    N = 100_000
    traces = Vector{UniversalTrace}(undef, N)
    lps = Vector{Float64}(undef, N)
    for i in 1:N
        old_model_trace = sample_trace(p1, ())
        old_proposal_trace = sample_trace(q1, (old_model_trace,))
        
        new_model_trace, new_proposal_trace = apply(f_transform, old_model_trace, old_proposal_trace)
        J = jacobian(f_transform, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)

        new_model_score = score_trace(p2, (), new_model_trace)
        new_proposal_score = score_trace(q2, (new_model_trace,), new_proposal_trace)

        lps[i] = new_model_score + new_proposal_score - old_model_score - old_proposal_score + logabsdet(J)[1]
        traces[i] = new_model_trace
    end
    UniversalTraces(traces, []), lps
end

using Plots
W = exp.(Evaluation.normalise(lps));

histogram(traces[:x], weights=W, normalize=true, lw=0, alpha=0.5, legend=false);
plot!(x -> exp(logpdf(Normal(0., 0.5),x)))

histogram(traces[:y], weights=W, normalize=true, lw=0, alpha=0.5, legend=false);
plot!(x -> exp(logpdf(Normal(1., 1.),x)))



# =========== Dirichlet transformation =================

@ppl function phi_model(α)
    K = length(α)
    α0 = sum(α)

    for j in 1:K-1
        α0 -= α[j] # sum_{i=j+1}^K α[i]
        {:phi => j} ~ Beta(α[j], α0)
    end
end

function to_dirchlet!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    K = length(old_trace) + 1

    s = 0. # sum_{i=1}^{j-1} x[i]
    for j in 1:K-1
        phi = read_continuous(tt, old_trace, :phi => j)
        x = (1-s)*phi
        s += x
        write_continuous(tt, new_trace, :x => j, x)
    end
end 

function from_dirchlet!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    K = length(old_trace) + 1

    s = 0. # sum_{i=1}^{j-1} x[i]
    for j in 1:K-1
        x = read_continuous(tt, old_trace, :x => j)
        phi = x / (1-s)
        s += x
        write_continuous(tt, new_trace, :phi => j, phi)
    end
end


transformation = TraceTransformation(to_dirchlet!)
inv_transformation = TraceTransformation(from_dirchlet!)

α = [1.,2.,1.,4.]

old_trace = sample_trace(phi_model, (α,))
new_trace = apply(transformation, old_trace)
J = jacobian(transformation, old_trace, new_trace)

# check inverse
old_trace_2 = apply(inv_transformation, new_trace)
all(old_trace[addr] ≈ old_trace_2[addr] for addr in keys(old_trace))
J_inv = jacobian(inv_transformation, new_trace, old_trace_2)
all(inv(J_inv) .≈ J)

# Dirichlet support lives on a K-1 dimensional manifold.
function embed(new_trace::UniversalTrace)
    K = length(new_trace) + 1
    x = [new_trace[:x => j] for j in 1:K-1]
    push!(x, 1 - sum(x))
    return x
end

x = embed(new_trace)

import Distributions: Dirichlet

logpdf(Dirichlet(α), x)
score_trace(phi_model, (α,), old_trace) - logabsdet(J)[1]