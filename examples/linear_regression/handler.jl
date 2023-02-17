
using TinyPPL.Distributions
using TinyPPL.Handlers
import DelimitedFiles
import Random
include("common.jl")

Random.seed!(7800)

function f(slope, intercept, x)
    intercept + slope * x
end

function LinReg(xs, ys)
    slope = sample(:slope, Normal(0.0, 10.))
    intercept = sample(:intercept, Normal(0.0, 10.))

    for i in 1:length(xs)
        sample((:y, i), Normal(f(slope, intercept, xs[i]), 1.), obs=ys[i])
    end

    return (slope, intercept)
end

const observations = Dict((:y, i) => y for (i, y) in enumerate(ys));

@info "likelihood_weighting"
traces, retvals, lps = likelihood_weighting(LinReg, (xs,ys), 100); # for compilation
@time traces, retvals, lps = likelihood_weighting(LinReg, (xs,ys), 10_000);
println("for 10_000 samples.")

W = exp.(lps);

slope_est = [r[1] for r in retvals]'W
intercept_est = [r[2] for r in retvals]'W

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, " (estimated) vs. ", map[2], " (map)")
println("  intercept: ", intercept_true, " vs. ", intercept_est, " (estimated) vs. ", map[1], " (map)")
