using TinyPPL.Distributions
using TinyPPL.Graph
import Random

# xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
xs = [-1., -0.5, 0.0, 0.5, 1.0];
ys = [-3.2, -1.8, -0.5, -0.2, 1.5];
N = length(xs)

function f(slope, intercept, x)
    intercept + slope * x
end

slope_prior_mean = 0
slope_prior_sigma = 3
intercept_prior_mean = 0
intercept_prior_sigma = 3

σ = 2.0
m0 = [0., 0.]
S0 = [intercept_prior_sigma^2 0.; 0. slope_prior_sigma^2]
Phi = hcat(fill(1., length(xs)), xs)
S = inv(inv(S0) + Phi'Phi / σ^2) 
map_mu = S*(inv(S0) * m0 + Phi'ys / σ^2)

map_mu
map_sigma = [sqrt(S[1,1]), sqrt(S[2,2])]


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

logjoint = Graph.make_logjoint(model)
K = sum(isnothing.(model.observed_values))


import TinyPPL: hmc_logjoint
Random.seed!(0)
result = hmc_logjoint(logjoint, K, 10_000, 10, 0.1)
mean(result,dims=2)

Random.seed!(0)
result = hmc(model, 10_000, 10, 0.1)
mean(result,dims=2)
