using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
ys = [-3.2, -1.8, -0.5, -0.2, 1.5];

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
map = S*(inv(S0) * m0 + Phi'ys / σ^2)

@ppl static function LinReg(xs)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)

    for i in 1:length(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

observations = Dict((:y, i) => y for (i, y) in enumerate(ys));
map
sqrt(S[1,1]), sqrt(S[2,2])

addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(LinReg, (xs,), observations);
K = length(addresses_to_ix)

Random.seed!(0)
mu, sigma = advi_meanfield(logjoint, 10_000, 10, 0.01, K)
Random.seed!(0)
mu, sigma = advi_meanfield(logjoint, 1_000_000, 10, 0.001, K)

Q = MeanFieldGaussian(K)
Random.seed!(0)
Q = advi(logjoint, 10_000, 10, 0.01, Q)



Random.seed!(0)
mu, L = advi_fullrank(logjoint, 10_000, 10, 0.01, K)
Random.seed!(0)
mu, L = advi_fullrank(logjoint, 100_000, 10, 0.001, K)

Q = MeanFieldGaussian(K)
Random.seed!(0)
Q = advi(logjoint, 10_000, 10, 0.01, Q)
Q = FullRankGaussian(K)