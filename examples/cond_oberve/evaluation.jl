using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

# https://okmij.org/ftp/kakuritu/Hakaru10/PPS2016.pdf
@ppl function pcond()
    x ~ Bernoulli(0.5)
    if x == 1
        return Int(x)
    else
        y ~ Bernoulli(0.5)
        z ~ Bernoulli(0.5)
        return z + 20
    end
end

Random.seed!(0)
traces, retvals, logprobs = likelihood_weighting(pcond, (), Dict(:y => 1), 1_000_000);

W = exp.(logprobs); sum(W)

sum(W[retvals .== 1])
sum(W[retvals .== 20])
sum(W[retvals .== 21])

Random.seed!(0)
traces, retvals, _ = lmh(pcond, (), Dict(:y => 1), 1_000_000);

mean(retvals .== 1)
mean(retvals .== 20)
mean(retvals .== 21)