
using TinyPPL.Graph
import Random
include("common.jl")

Random.seed!(7800)

const model = @ppl LinReg begin
    function get_xs()
        [
            1., 2., 3, 4., 5., 6., 7., 8., 9., 10.,
            11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
            21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
            31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
            41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
            51., 52., 53., 54., 55., 56., 57., 58., 59., 60.,
            61., 62., 63., 64., 65., 66., 67., 68., 69., 70.,
            71., 72., 73., 74., 75., 76., 77., 78., 79., 80.,
            81., 82., 83., 84., 85., 86., 87., 88., 89., 90.,
            91., 92., 93., 94., 95., 96., 97., 98., 99., 100.
        ]
    end
    function get_ys()
        [
            0.77, 3.94, 5.6, 9.0, 8.95,
            12.17, 11.3, 12.88, 17.56, 18.14,
            20.73, 23.58, 26.16, 27.29, 28.56,
            31.13, 33.17, 33.93, 35.74, 39.2,
            39.93, 41.62, 45.95, 46.71, 49.33,
            50.5, 53.22, 53.34, 58.93, 57.95,
            62.72, 62.33, 65.26, 67.64, 68.72,
            71.02, 71.97, 75.46, 77.28, 79.45,
            80.48, 81.44, 84.04, 88.26, 87.63,
            90.74, 94.25, 94.74, 98.32, 98.9,
            102.86, 103.23, 103.37, 107.81, 109.07,
            111.13, 112.1, 115.76, 117.86, 118.46,
            120.42, 122.09, 123.63, 127.66, 129.04,
            131.88, 134.79, 133.64, 135.52, 139.48,
            141.27, 144.24, 146.11, 148.85, 148.87,
            150.59, 152.62, 154.5, 156.36, 161.28,
            161.85, 162.79, 164.86, 166.88, 169.47,
            170.91, 172.45, 174.53, 177.52, 178.31,
            181.65, 182.27, 184.36, 187.1, 190.0,
            191.64, 193.24, 195.1, 196.71, 199.2
        ]
    end

    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = get_xs(),
        ys = get_ys(),
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:100]
        
        (slope, intercept)
    end
end

const slope_true = 2.
const intercept_true = -1.

@info "likelihood_weighting"
traces, retvals, lps = likelihood_weighting(model, 100); # for compilation
@time traces, retvals, lps = likelihood_weighting(model, 100_000);
println("for 100_000 samples.")

W = exp.(lps);

slope_est = [r[1] for r in retvals]'W
intercept_est = [r[2] for r in retvals]'W

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, " (estimated)")
println("  intercept: ", intercept_true, " vs. ", intercept_est, " (estimated)")


@info "HMC"
@time traces, retvals, lps = hmc(model, 1_000, 0.001, 10, [1. 0.; 0. 1.]);
println("for 10_000 samples.")

slope_est = mean([r[1] for r in retvals])
intercept_est = mean([r[2] for r in retvals])

println("convergence:")
println("  slope: ", slope_true, " vs. ", slope_est, " (estimated)")
println("  intercept: ", intercept_true, " vs. ", intercept_est, " (estimated)")
