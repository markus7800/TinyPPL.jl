# Trace transformations: differentiable maps between traces, with automatically computed Jacobians.
#
# A TraceTransformation is defined by a function that reads values from old traces and writes values to new traces
# with read_continuous / write_continuous (and read_discrete / write_discrete for discrete values).
# `apply` runs it and `jacobian` computes the Jacobian of the continuous part with automatic differentiation.
# By the change of variables formula, if the old trace has density p, the new trace has density p - log|det J| (log scale).
# Trace transformations are the building block of involutive MCMC, see imcmc.jl.
#
# Run: julia --project=examples examples/universal/trace_transform.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import LinearAlgebra: logabsdet
import TinyPPL: logsumexp

# ===== Box-Muller transform =====
# Maps two independent Uniform(0, 1) variables to two independent standard normal variables.
function box_muller!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    u1 = read_continuous(tt, old_trace, :u1)
    u2 = read_continuous(tt, old_trace, :u2)
    r = sqrt(-2 * log(u1))
    θ = 2π * u2
    write_continuous(tt, new_trace, :x, r * cos(θ))
    write_continuous(tt, new_trace, :y, r * sin(θ))
end

function box_muller_inv!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    x = read_continuous(tt, old_trace, :x)
    y = read_continuous(tt, old_trace, :y)
    write_continuous(tt, new_trace, :u1, exp(-(x^2 + y^2) / 2))
    write_continuous(tt, new_trace, :u2, mod(atan(y, x), 2π) / 2π)
end

transformation = TraceTransformation(box_muller!)
inv_transformation = TraceTransformation(box_muller_inv!)

old_trace = UniversalTrace(:u1 => 0.5, :u2 => 0.3)
new_trace = apply(transformation, old_trace)
J = jacobian(transformation, old_trace, new_trace)
println("Box-Muller: ", old_trace, " ↦ ", new_trace)

old_trace_2 = apply(inv_transformation, new_trace)
J_inv = jacobian(inv_transformation, new_trace, old_trace_2)
println("inverse recovers the old trace: ", all(old_trace[addr] ≈ old_trace_2[addr] for addr in keys(old_trace)),
    ", Jacobian of the inverse is the inverse Jacobian: ", inv(J_inv) ≈ J)

# The density of the uniforms is 1, so the density of (x, y) is 0 - log|det J|, which equals the standard normal density.
println("log p(x, y) by change of variables: ", logpdf(Uniform(0, 1), old_trace[:u1]) + logpdf(Uniform(0, 1), old_trace[:u2]) - logabsdet(J)[1],
    ", standard normal: ", logpdf(Normal(0, 1), new_trace[:x]) + logpdf(Normal(0, 1), new_trace[:y]))


# ===== Transformations between models of different type, with auxiliary variables =====
# p1 samples two discrete variables i, j ∈ 1:10, q1 samples continuous offsets dx, dy ∈ [0, 0.1].
# u1 = (i-1)/10 + dx and u2 = (j-1)/10 + dy are Uniform(0, 1), and Box-Muller maps them to (x, y).
# The inverse recovers i, j, dx, dy from (x, y). The discrete part does not contribute to the Jacobian.
@ppl function p1()
    i ~ DiscreteUniform(1, 10)
    j ~ DiscreteUniform(1, 10)
end

@ppl function q1(p1_trace)
    dx ~ Uniform(0., 0.1)
    dy ~ Uniform(0., 0.1)
end

function f!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    i = read_discrete(tt, old_model_trace, :i)
    j = read_discrete(tt, old_model_trace, :j)
    dx = read_continuous(tt, old_proposal_trace, :dx)
    dy = read_continuous(tt, old_proposal_trace, :dy)

    r = sqrt(-2 * log((i - 1) / 10 + dx))
    θ = 2π * ((j - 1) / 10 + dy)
    write_continuous(tt, new_model_trace, :x, r * cos(θ))
    write_continuous(tt, new_model_trace, :y, r * sin(θ))
end

function finv!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    x = read_continuous(tt, old_model_trace, :x)
    y = read_continuous(tt, old_model_trace, :y)
    u1 = exp(-(x^2 + y^2) / 2)
    u2 = mod(atan(y, x), 2π) / 2π
    i = ceil(u1 * 10)
    j = ceil(u2 * 10)
    write_discrete(tt, new_model_trace, :i, i)
    write_discrete(tt, new_model_trace, :j, j)
    write_continuous(tt, new_proposal_trace, :dx, u1 - (i - 1) / 10)
    write_continuous(tt, new_proposal_trace, :dy, u2 - (j - 1) / 10)
end

f_transform = TraceTransformation(f!)
finv_transform = TraceTransformation(finv!)

Random.seed!(0)
old_model_trace = sample_trace(p1, ())
old_proposal_trace = sample_trace(q1, (old_model_trace,))
new_model_trace, new_proposal_trace = apply(f_transform, old_model_trace, old_proposal_trace)
J = jacobian(f_transform, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)

old_model_trace_2, old_proposal_trace_2 = apply(finv_transform, new_model_trace, new_proposal_trace)
println("\ninverse recovers the old traces: ",
    all(old_model_trace[addr] ≈ old_model_trace_2[addr] for addr in keys(old_model_trace)) &&
    all(old_proposal_trace[addr] ≈ old_proposal_trace_2[addr] for addr in keys(old_proposal_trace)))

# The transformed traces are again standard normal (p1 and q1 have densities 1/100 and 100).
old_score = score_trace(p1, (), old_model_trace) + score_trace(q1, (old_model_trace,), old_proposal_trace)
println("log p(x, y) by change of variables: ", old_score - logabsdet(J)[1],
    ", standard normal: ", logpdf(Normal(0, 1), new_model_trace[:x]) + logpdf(Normal(0, 1), new_model_trace[:y]))

# Thus f gives a proposal for any model p2 over (x, y), here with the empty auxiliary model q2.
# We correct for the difference between the standard normal and p2 with the importance weight
#   w = p2(x, y) q2() / (p1(i, j) q1(dx, dy)) * |det J|.
@ppl function p2()
    x ~ Normal(0., 0.5)
    y ~ Normal(1., 1.)
end

@ppl function q2(p2_trace)
end

Random.seed!(0)
N = 100_000
xs, ys, logw = zeros(N), zeros(N), zeros(N)
for n in 1:N
    local old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, J
    old_model_trace = sample_trace(p1, ())
    old_proposal_trace = sample_trace(q1, (old_model_trace,))
    new_model_trace, new_proposal_trace = apply(f_transform, old_model_trace, old_proposal_trace)
    J = jacobian(f_transform, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
    logw[n] = score_trace(p2, (), new_model_trace) + score_trace(q2, (new_model_trace,), new_proposal_trace) -
              score_trace(p1, (), old_model_trace) - score_trace(q1, (old_model_trace,), old_proposal_trace) + logabsdet(J)[1]
    xs[n], ys[n] = new_model_trace[:x], new_model_trace[:y]
end
W = exp.(logw .- logsumexp(logw))
weighted_std(v) = sqrt(W' * (v .- W'v) .^ 2)
println("importance sampling p2: mean and std of x ≈ ", round.((W'xs, weighted_std(xs)), digits=3), " (exact: (0.0, 0.5)), ",
    "of y ≈ ", round.((W'ys, weighted_std(ys)), digits=3), " (exact: (1.0, 1.0))")


# ===== Stick-breaking construction of the Dirichlet distribution =====
# If phi_j ~ Beta(α_j, α_{j+1} + ... + α_K) independently, then x_j = (1 - x_1 - ... - x_{j-1}) phi_j for j < K
# and x_K = 1 - x_1 - ... - x_{K-1} is Dirichlet(α) distributed. The Dirichlet lives on the (K-1)-dimensional simplex,
# so the transformation only writes x_1, ..., x_{K-1}.
@ppl function stick_breaking(α)
    K = length(α)
    α_rest = sum(α)
    for j in 1:K-1
        α_rest -= α[j]
        {:phi => j} ~ Beta(α[j], α_rest)
    end
end

function to_dirichlet!(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    s = 0. # x_1 + ... + x_{j-1}
    for j in 1:length(old_trace)
        x = (1 - s) * read_continuous(tt, old_trace, :phi => j)
        s += x
        write_continuous(tt, new_trace, :x => j, x)
    end
end

α = [1., 2., 1., 4.]
transformation = TraceTransformation(to_dirichlet!)

Random.seed!(0)
old_trace = sample_trace(stick_breaking, (α,))
new_trace = apply(transformation, old_trace)
J = jacobian(transformation, old_trace, new_trace)
x = [new_trace[:x => j] for j in 1:length(α)-1]
push!(x, 1 - sum(x))

import Distributions: Dirichlet
println("\nlog p(x) by change of variables: ", score_trace(stick_breaking, (α,), old_trace) - logabsdet(J)[1],
    ", Dirichlet: ", logpdf(Dirichlet(α), x))
