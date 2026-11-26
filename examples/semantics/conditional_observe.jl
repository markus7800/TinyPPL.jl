# Observations inside branches (Kiselyov, PPS 2016: https://okmij.org/ftp/kakuritu/Hakaru10/PPS2016.pdf).
#
# The variable y only exists if x = 0, and we observe y = 1. In the branch x = 1 there is nothing to observe,
# so its executions keep weight 1, while executions with x = 0 get weight P(y = 1) = 0.5.
# Thus the observation shifts the posterior towards x = 1, although y carries no information about x:
#   P(x = 1 | y = 1) = 0.5 / (0.5 + 0.5 * 0.5) = 2/3, and the return values 20 and 21 have probability 1/6 each.
# This is the semantics of both universal and graph-based models (an observed node in a branch that is not taken
# gets a Flat distribution, see the printed model).
#
# Run: julia --project=examples examples/semantics/conditional_observe.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
using TinyPPL.Graph
import Random

@ppl function pcond()
    x ~ Bernoulli(0.5)
    if x == 1
        return x
    else
        y ~ Bernoulli(0.5)
        z ~ Bernoulli(0.5)
        return z + 20
    end
end

pcond_pgm = @pgm pcond begin
    let x ~ Bernoulli(0.5)
        if x == 1
            x
        else
            let z ~ Bernoulli(0.5)
                {:y} ~ Bernoulli(0.5) ↦ 1.
                z + 20
            end
        end
    end
end
println(pcond_pgm)

# Both modules export likelihood_weighting and lmh, so we qualify them.
observations = Observations(:y => 1)
print_probabilities(method, values, W=fill(1 / length(values), length(values))) =
    println(rpad(method, 32), "P(1) ≈ ", round(sum(W[values .== 1]), digits=3),
        ", P(20) ≈ ", round(sum(W[values .== 20]), digits=3), ", P(21) ≈ ", round(sum(W[values .== 21]), digits=3))

println("exact                           P(1) = ", round(2/3, digits=3), ", P(20) = ", round(1/6, digits=3), ", P(21) = ", round(1/6, digits=3))

Random.seed!(0)
traces, logw = Evaluation.likelihood_weighting(pcond, (), observations, 1_000_000)
print_probabilities("universal, likelihood weighting", retvals(traces), exp.(logw))

Random.seed!(0)
traces = Evaluation.lmh(pcond, (), observations, 1_000_000)
print_probabilities("universal, LMH", retvals(traces))

Random.seed!(0)
traces, logw = Graph.likelihood_weighting(pcond_pgm, 1_000_000)
print_probabilities("graph, likelihood weighting", retvals(traces), exp.(logw))

Random.seed!(0)
traces = Graph.lmh(pcond_pgm, 1_000_000)
print_probabilities("graph, LMH", retvals(traces))
