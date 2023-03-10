
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("common.jl")

Random.seed!(7800)

@ppl function geometric(p::Float64)
    i = 0
    while true
        b = {(:b,i)} ~ Bernoulli(p)
        b && break
        i += 1
    end
    {:X} ~ Normal(i, 1.0)
    return i
end

const p = 0.5;
const observations = Dict(:X => 5);
posterior_true = get_true_posterior(p, 10, observations[:X])

@info "likelihood_weighting"
traces, retvals, lps = likelihood_weighting(geometric, (p,), observations, 1_000); # for compilation
@time traces, retvals, lps = likelihood_weighting(geometric, (p,), observations, 1_000_000);
println("for 1_000_000 samples.")

W = exp.(lps);
posterior_est = [sum(W[retvals .== i]) for i in 0:10]
posterior_diff = maximum(abs.(posterior_true .- posterior_est))

println("convergence:")
println("  difference to true prosterior: ", posterior_diff)
