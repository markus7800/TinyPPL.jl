using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
xs = [-1., -0.5, 0.0, 0.5, 1.0];
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
map_mu = S*(inv(S0) * m0 + Phi'ys / σ^2)

map_mu
map_sigma = [sqrt(S[1,1]), sqrt(S[2,2])]

@ppl static function LinReg(xs)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

observations = Dict((:y, i) => y for (i, y) in enumerate(ys));

addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(LinReg, (xs,), observations);
K = length(addresses_to_ix)

Random.seed!(0)
mu, sigma = advi_meanfield(logjoint, 10_000, 10, 0.01, K)
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)

#Random.seed!(0)
#mu, sigma = advi_meanfield(logjoint, 100_000, 100, 0.001, K)

Random.seed!(0)
Q = advi(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())
maximum(abs, Q.mu .- map_mu)
maximum(abs, Q.sigma .- map_sigma)

Random.seed!(0)
Q2 = advi(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO());

# TODO: mathetmatically check why these are the same
maximum(abs, Q2.mu .- Q.mu)
maximum(abs, Q2.sigma .- Q.sigma)

N = 10_000
Random.seed!(0)
mu, L = advi_fullrank(logjoint, N, 10, 0.01, K);
Random.seed!(0)
Q = advi(logjoint, N, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO());
Random.seed!(0)
Q2 = advi(logjoint, N, 10, 0.01, FullRankGaussian(K), MonteCarloELBO());

maximum(abs, mu .- Q.base.μ)
maximum(abs, L*L' .- Q.base.Σ)

# TODO: mathetmatically check why these are the same
maximum(abs, Q2.base.μ .- Q.base.μ)
maximum(abs, Q2.base.Σ .- Q.base.Σ)

Random.seed!(0)
mu, L = advi_fullrank(logjoint, 10_000, 100, 0.01, K)
L*L'


@ppl static function LinRegGuide()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = exp(param("omega_intercept"))
    sigma2 = exp(param("omega_slope"))

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end

# @ppl static function FullRankLinRegGuide()
#     mu = param("mu", 2)
#     L = param("L", 4)
#     zeta = {:zeta} ~ MvNormal(mu, L*L') # TODO

#     {:intercept} ~ Dirac(zeta[1])
#     {:slope} ~ Dirac(zeta[2])
# end

Random.seed!(0);
@time Q = advi(logjoint, 10_000, 100, 0.01, MeanFieldGaussian(K), MonteCarloELBO())

guide = make_guide(LinRegGuide, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time Q2 = advi(logjoint, 10_000, 100, 0.01, guide, MonteCarloELBO())
mu = vcat(Q2.sampler.phi[Q2.sampler.params_to_ix["mu_intercept"]], Q2.sampler.phi[Q2.sampler.params_to_ix["mu_slope"]])
sigma = exp.(vcat(Q2.sampler.phi[Q2.sampler.params_to_ix["omega_intercept"]], Q2.sampler.phi[Q2.sampler.params_to_ix["omega_slope"]]))

maximum(abs, mu .- Q.mu)
maximum(abs, sigma .- Q.sigma)