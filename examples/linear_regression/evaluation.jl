
using TinyPPL.Distributions
using TinyPPL.Evaluation
import DelimitedFiles
import Random
include("common.jl")

Random.seed!(7800)

function f(slope, intercept, x)
    intercept + slope * x
end

@ppl function LinReg(xs)
    slope = {:slope} ~ Normal(0.0, 10.)
    intercept = {:intercept} ~ Normal(0.0, 10.)

    for i in 1:length(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), 1.)
    end

    return (slope, intercept)
end

const slope_true = 2.
const intercept_true = -1.
const xs = vec(DelimitedFiles.readdlm("data/xs.txt"))
const ys = vec(DelimitedFiles.readdlm("data/ys.txt"))
const observations = Dict((:y, i) => y for (i, y) in enumerate(ys));

@info "likelihood_weighting"
traces, retvals, lps = likelihood_weighting(LinReg, (xs,), observations, 100); # for compilation
@time traces, retvals, lps = likelihood_weighting(LinReg, (xs,), observations, 1_000_000);
println("for 1_000_000 samples.")

W = exp.(lps);

slope_est = [r[1] for r in retvals]'W
intercept_est = [r[2] for r in retvals]'W

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, " (estimated)")
println("  intercept: ", intercept_true, " vs. ", intercept_est, " (estimated)")
