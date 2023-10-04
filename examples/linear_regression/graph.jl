
using TinyPPL.Distributions: mean
using TinyPPL.Graph
import Random
import DelimitedFiles
include("common.jl")

Random.seed!(7800)

const model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = $(Main.xs),
        ys = $(Main.ys),
        slope ~ Normal($(Main.slope_prior_mean), $(Main.slope_prior_sigma)),
        intercept ~ Normal($(Main.intercept_prior_mean), $(Main.intercept_prior_sigma))

        [(Normal(f(slope, intercept, xs[i]), $(Main.σ)) ↦ ys[i]) for i in 1:$(Main.N)]
        
        (slope, intercept)
    end
end

@info "likelihood_weighting"
traces, retvals, lps = likelihood_weighting(model, 100); # for compilation
@time traces, retvals, lps = likelihood_weighting(model, 100_000);
println("for 100_000 samples.")

W = exp.(lps);

slope_est = [r[1] for r in retvals]'W
intercept_est = [r[2] for r in retvals]'W

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, " (estimated) vs. ", map[2], " (map)")
println("  intercept: ", intercept_true, " vs. ", intercept_est, " (estimated) vs. ", map[1], " (map)")


@info "compiled_likelihood_weighting"
const lw = compile_likelihood_weighting(model)


traces, retvals, lps = compiled_likelihood_weighting(model, lw, 100; static_observes=true); # for compilation

@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 10_000_000; static_observes=true);
println("for 10_000_000 samples.")

W = exp.(lps);

slope_est = [r[1] for r in retvals]'W
slope_sigma_est = sqrt([r[1]^2 for r in retvals]'W - slope_est^2)
intercept_est = [r[2] for r in retvals]'W
intercept_sigma_est = sqrt([r[2]^2 for r in retvals]'W - intercept_est^2)

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, "±", slope_sigma_est, " (estimated) vs. ", map[2], "±", sqrt(S[2,2]), " (map)")
println("  intercept: ", intercept_true," vs. ", intercept_est, "±", intercept_sigma_est, " (estimated) vs. ", map[1], "±", sqrt(S[1,1]), " (map)")



@info "HMC"
traces, retvals, lps = hmc(model, 10, 0.001, 10, [1. 0.; 0. 1.]); # for compilation
@time traces, retvals, lps = hmc(model, 1_000, 0.001, 10, [1. 0.; 0. 1.]);
println("for 10_000 samples.")

slope_est = mean([r[1] for r in retvals])
intercept_est = mean([r[2] for r in retvals])

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, " (estimated) vs. ", map[2], " (map)")
println("  intercept: ", intercept_true, " vs. ", intercept_est, " (estimated) vs. ", map[1], " (map)")
