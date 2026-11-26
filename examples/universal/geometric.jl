# A geometric distribution written as a while loop and as recursion.
#
# Flip a coin with success probability p until the first success; i is the number of failures, so i ~ Geometric(p).
# We observe X ~ Normal(i, 1) and infer i. The number of coin flips is unbounded, which needs a universal model.
# The second version shows recursion with @subppl: the called model's sample statements become part of the trace.
#
# Run: julia --project=examples examples/universal/geometric.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import TinyPPL: logsumexp

@ppl function geometric_loop(p::Float64)
    i = 0
    while true
        b = {:b => i} ~ Bernoulli(p)
        b && break
        i += 1
    end
    X ~ Normal(i, 1.)
    return i
end

@ppl function geometric_recursion(p::Float64, i::Int)
    b = {:b => i} ~ Bernoulli(p)
    if b
        X ~ Normal(i, 1.)
        return i
    else
        return @subppl geometric_recursion(p, i + 1)
    end
end

p = 0.5
observations = Observations(:X => 5.)

# p(i | X) ∝ Geometric(i; p) * Normal(X; i, 1), normalised over i = 0:250 (the remaining mass is negligible).
log_joint = [logpdf(Geometric(p), i) + logpdf(Normal(i, 1.), observations[:X]) for i in 0:250]
exact = exp.(log_joint .- logsumexp(log_joint))[1:11]

Random.seed!(0)
traces, logw = likelihood_weighting(geometric_loop, (p,), observations, 1_000_000)
W = exp.(logw)
lw_loop = [sum(W[retvals(traces) .== i]) for i in 0:10]

# Both models produce the same traces (addresses :b => 0, :b => 1, ..., and :X), so with the same seed
# the results are identical.
Random.seed!(0)
traces, logw = likelihood_weighting(geometric_recursion, (p, 0), observations, 1_000_000)
W = exp.(logw)
lw_recursion = [sum(W[retvals(traces) .== i]) for i in 0:10]

println("P(i | X = 5) for i = 0:10")
println("  exact                    ", round.(exact, digits=3))
println("  likelihood weighting     ", round.(lw_loop, digits=3))
println("  ... with recursive model ", round.(lw_recursion, digits=3))
