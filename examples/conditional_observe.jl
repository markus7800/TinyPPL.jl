using TinyPPL.Distributions
using TinyPPL.Evaluation
using TinyPPL.Graph
import Random

# https://okmij.org/ftp/kakuritu/Hakaru10/PPS2016.pdf

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
args = ()
observations = Observations(:y => 1)

Random.seed!(0)
traces, logprobs = Evaluation.likelihood_weighting(pcond, args, observations, 1_000_000);

W = exp.(logprobs);

sum(W[retvals(traces) .== 1])
sum(W[retvals(traces) .== 20])
sum(W[retvals(traces) .== 21])

Random.seed!(0)
traces = Evaluation.lmh(pcond, args, observations, 1_000_000);

mean(retvals(traces) .== 1)
mean(retvals(traces) .== 20)
mean(retvals(traces) .== 21)

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


Random.seed!(0)
traces, logprobs = Graph.likelihood_weighting(pcond_pgm, 1_000_000);

W = exp.(logprobs);

sum(W[retvals(traces) .== 1])
sum(W[retvals(traces) .== 20])
sum(W[retvals(traces) .== 21])

Random.seed!(0)
traces = Graph.lmh(pcond_pgm, 1_000_000);

mean(retvals(traces) .== 1)
mean(retvals(traces) .== 20)
mean(retvals(traces) .== 21)
