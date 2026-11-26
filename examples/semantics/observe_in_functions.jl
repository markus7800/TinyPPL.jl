# Observations inside functions are not compositional
# (Holtzen, Van den Broeck, Millstein: Scaling Exact Inference for Discrete Probabilistic Programs, OOPSLA 2020, Section 3.2.2).
#
# The functions f and g below both always return 1, but replacing one by the other changes the posterior,
# because f contains an observation, which reweights the executions of its caller.
# So a function cannot be described by its return value distribution alone.
#
# Run: julia --project=examples examples/semantics/observe_in_functions.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import Random

# Observing a deterministic expression: `Dirac(value) ↦ 1` conditions on value == 1.
# P(x = 1 | x or y) = P(x = 1) / P(x or y) = 0.6 / (1 - 0.4 * 0.7)
model = @pgm or_model begin
    function or(x, y)
        max(x, y)
    end
    let x ~ Bernoulli(0.6),
        y ~ Bernoulli(0.3)

        Dirac(or(x, y)) ↦ 1
        x
    end
end

Random.seed!(0)
traces, logw = likelihood_weighting(model, 1_000_000)
# P(X = 1 | X || Y = 1)  = P( X = 1,  X || Y = 1) / P(X || Y = 1) = P(X = 1) / (1 - P(X=0, Y=0))
println("P(x = 1 | x or y) ≈ ", round(retvals(traces)'exp.(logw), digits=4), " (exact: ", round(0.6 / (1 - 0.4 * 0.7), digits=4), ")")

# f(x) flips a coin and observes that x or flip is true, so it always returns 1.
# The observation rules out x = 0 and flip = 0: P(x = 1 | f) = 0.1 / (1 - 0.9 * 0.5).
f_model = @pgm f_model begin
    function or(x, y)
        max(x, y)
    end
    function f(x)
        let flip ~ Bernoulli(0.5),
            y = or(x, flip)

            Dirac(y) ↦ 1
            y
        end
    end
    let x ~ Bernoulli(0.1),
        obs = f(x)

        x
    end
end

Random.seed!(0)
traces, logw = likelihood_weighting(f_model, 1_000_000)
println("P(x = 1) with f ≈ ", round(retvals(traces)'exp.(logw), digits=4), " (exact: ", round(0.1 / (1 - 0.9 * 0.5), digits=4), ")")

# g(x) also always returns 1, but does not observe anything, so the posterior is the prior.
g_model = @pgm g_model begin
    function g(x)
        1
    end
    let x ~ Bernoulli(0.1),
        obs = g(x)

        x
    end
end

Random.seed!(0)
traces, logw = likelihood_weighting(g_model, 1_000_000)
println("P(x = 1) with g ≈ ", round(retvals(traces)'exp.(logw), digits=4), " (exact: 0.1)")
