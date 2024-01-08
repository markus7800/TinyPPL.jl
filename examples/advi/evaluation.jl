using TinyPPL.Distributions
using TinyPPL.Evaluation
using TinyPPL.Logjoint
import Random

# xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
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
map_Σ = inv(inv(S0) + Phi'Phi / σ^2) 
map_mu = map_Σ*(inv(S0) * m0 + Phi'ys / σ^2)
map_sigma = [sqrt(map_Σ[1,1]), sqrt(map_Σ[2,2])]

@ppl static function LinRegStatic(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)
    # println("intercept: ", intercept, ", slope: ", slope)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

observations = Observations((:y, i) => y for (i, y) in enumerate(ys));
ulj = Evaluation.make_unconstrained_logjoint(LinRegStatic, (xs,), observations)
addresses_to_ix = get_address_to_ix(LinRegStatic, (xs,), observations)
K = length(addresses_to_ix)

Random.seed!(0)
traces = hmc(LinRegStatic, (xs,), observations, 10_000, 10, 0.1)

plot(traces[:slope]);
plot!(traces[:intercept])

maximum(abs, mean(traces[:intercept]) - map_mu[1])
maximum(abs, mean(traces[:slope]) - map_mu[2])

Random.seed!(0)
vi_result = advi_meanfield(LinRegStatic, (xs,), observations, 10_000, 10, 0.01);
maximum(abs, vi_result.Q.mu .- map_mu)
maximum(abs, vi_result.Q.sigma .- map_sigma)

Random.seed!(0)
mu_lj, sigma_lj = advi_meanfield_logjoint(ulj.logjoint, K, 10_000, 10, 0.01)
maximum(abs, vi_result.Q.mu .- mu_lj)
maximum(abs, vi_result.Q.sigma .- sigma_lj)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())
# equivalent to advi_meanfield_logjoint
maximum(abs, vi_result_2.Q.mu .- mu_lj)
maximum(abs, vi_result_2.Q.sigma .- sigma_lj)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO());
# equivalent to MonteCarloELBO because ∇ log Q = ∇ entropy
maximum(abs, vi_result_2.Q.mu .- vi_result.Q.mu)
maximum(abs, vi_result_2.Q.sigma .- vi_result.Q.sigma)

Random.seed!(0)
vi_result_3 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanField([VariationalNormal(), VariationalNormal()]), MonteCarloELBO())
Q3_mu = [d.base.μ for d in vi_result_3.Q.dists]
Q3_sigma = [d.base.σ for d in vi_result_3.Q.dists]
# equivalent to MeanFieldGaussian + MonteCarloELBO()
maximum(abs, vi_result_2.Q.mu .- Q3_mu)
maximum(abs, vi_result_2.Q.sigma .- Q3_sigma)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), ReinforceELBO())
maximum(abs, vi_result_2.Q.mu .- map_mu)
maximum(abs, vi_result_2.Q.sigma .- map_sigma)

# bbvi is different to advi
maximum(abs, vi_result_2.Q.mu .- vi_result.Q.mu)
maximum(abs, vi_result_2.Q.sigma .- vi_result.Q.sigma)


Random.seed!(0)
q = get_mixed_meanfield(LinRegStatic, (xs,), observations, ulj.addresses_to_ix)
vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, q, ReinforceELBO());
Q2_mu = [d.base.μ for d in vi_result_2.Q.dists]
Q2_sigma = [d.base.σ for d in vi_result_2.Q.dists]

Random.seed!(0)
vi_result_3 = bbvi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01)
Q3_mu = [d.base.μ for d in vi_result_3.Q.dists]
Q3_sigma = [d.base.σ for d in vi_result_3.Q.dists]
# equivalent to MeanField + ReinforceELBO()
maximum(abs, Q2_mu .- Q3_mu)
maximum(abs, Q2_sigma .- Q3_sigma)


Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), PathDerivativeELBO())
# PathDerivativeELBO is best
maximum(abs, vi_result_2.Q.mu .- map_mu)
maximum(abs, vi_result_2.Q.sigma .- map_sigma)

# pd elbo is different to advi
maximum(abs, vi_result_2.Q.mu .- vi_result.Q.mu)
maximum(abs, vi_result_2.Q.sigma .- vi_result.Q.sigma)


N = 10_000
Random.seed!(0)
vi_result = advi_fullrank(LinRegStatic, (xs,), observations, N, 10, 0.01);
maximum(abs, map_mu .- vi_result.Q.base.μ)
maximum(abs, map_Σ .- vi_result.Q.base.Σ)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, N, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO());
# equivalent to advi_fullrank_logjoint
maximum(abs, vi_result.Q.base.μ .- vi_result_2.Q.base.μ)
maximum(abs, vi_result.Q.base.Σ .- vi_result_2.Q.base.Σ)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, N, 10, 0.01, FullRankGaussian(K), MonteCarloELBO());
# equivalent to RelativeEntropyELBO because ∇ log Q = ∇ entropy
maximum(abs, vi_result.Q.base.μ .- vi_result_2.Q.base.μ)
maximum(abs, vi_result.Q.base.Σ .- vi_result_2.Q.base.Σ)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, N, 100, 0.01, FullRankGaussian(K), ReinforceELBO());
maximum(abs, map_mu .- vi_result_2.Q.base.μ)
maximum(abs, map_Σ .- vi_result_2.Q.base.Σ)

# bbvi is differnt to advi
maximum(abs, vi_result_2.Q.base.μ .- vi_result.Q.base.μ)
maximum(abs, vi_result_2.Q.base.Σ .- vi_result.Q.base.Σ)

Random.seed!(0)
vi_result_2 = advi(LinRegStatic, (xs,), observations, N, 10, 0.01, FullRankGaussian(K), PathDerivativeELBO());
# PathDerivativeELBO is best
maximum(abs, map_mu .- vi_result_2.Q.base.μ)
maximum(abs, map_Σ .- vi_result_2.Q.base.Σ)

# but different
maximum(abs, vi_result_2.Q.base.μ .- vi_result.Q.base.μ)
maximum(abs, vi_result_2.Q.base.Σ .- vi_result.Q.base.Σ)


@ppl static function LinRegGuideStatic()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = exp(param("omega_intercept"))
    sigma2 = exp(param("omega_slope"))

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end

@ppl static function LinRegGuideStatic2()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = param("sigma_intercept", 1, Positive())
    sigma2 = param("sigma_slope", 1, Positive())

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
@time vi_result = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO())

guide = make_guide(LinRegGuideStatic, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, guide, MonteCarloELBO())

Random.seed!(0)
@time vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, LinRegGuideStatic, (), MonteCarloELBO())

parameters = get_constrained_parameters(vi_result_2.Q)
mu = vcat(parameters["mu_intercept"], parameters["mu_slope"])
sigma = exp.(vcat(parameters["omega_intercept"], parameters["omega_slope"]))
# equivalent to advi_logjoint
maximum(abs, mu .- vi_result.Q.mu)
maximum(abs, sigma .- vi_result.Q.sigma)



Random.seed!(0)
@time vi_result_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, LinRegGuideStatic2, (), MonteCarloELBO())
parameters = get_constrained_parameters(vi_result_2.Q)

mu = vcat(parameters["mu_intercept"], parameters["mu_slope"])
sigma = vcat(parameters["sigma_intercept"], parameters["sigma_slope"])
# equivalent to advi_logjoint
maximum(abs, mu .- vi_result.Q.mu)
maximum(abs, sigma .- vi_result.Q.sigma)


using Plots
@ppl static function unif()
    x ~ Uniform(-1,1)
    y ~ Uniform(x-1,x+1)
    z ~ Uniform(y-1,y+1)
end

Random.seed!(0)
traces, retvals, lp = likelihood_weighting(unif, (), Dict(), 1_000_000);
histogram(traces[:y], weights=exp.(lp), normalize=true, legend=false)

Random.seed!(0)
@time traces = hmc(unif, (), Dict(), 100_000, 10, 0.1; ad_backend=:reversediff);
# tracker: 25.944568 seconds (653.12 M allocations: 20.248 GiB, 18.38% gc time)
# forwarddiff: 9.522213 seconds (205.32 M allocations: 8.314 GiB, 17.19% gc time)
# reversediff: 27.276258 seconds (553.02 M allocations: 20.625 GiB, 14.25% gc time)
histogram(traces[:y], normalize=true, legend=false)

addresses_to_ix = get_address_to_ix(unif, (), Dict())
K = length(addresses_to_ix)

Random.seed!(0)
vi_result = advi(unif, (), Dict(), 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())
vi_result = advi(unif, (), Dict(), 10_000, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO())
posterior = sample_posterior(vi_result, 1_000_000)
histogram(posterior[:y], normalize=true, legend=false)




@ppl function LinReg(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

@ppl function LinRegGuide()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = param("sigma_intercept", 1, Positive())
    sigma2 = param("sigma_slope", 1, Positive())

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end

observations = Observations((:y, i) => y for (i, y) in enumerate(ys));

Random.seed!(0)
vi_result = advi_meanfield(LinReg, (xs,), observations,  10_000, 10, 0.01)
mu = [vi_result.Q[:intercept].base.μ, vi_result.Q[:slope].base.μ]
sigma = [vi_result.Q[:intercept].base.σ, vi_result.Q[:slope].base.σ]
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)

Random.seed!(0)
vi_result_2 = advi(LinReg, (xs,), observations,  10_000, 10, 0.01, LinRegGuide, (), MonteCarloELBO());
parameters = get_constrained_parameters(vi_result_2.Q)


Random.seed!(0)
vi_result_2 = advi(LinReg, (xs,), observations,  10_000, 10, 0.01, LinRegGuide, (), PathDerivativeELBO());

Random.seed!(0)
vi_result_2 = advi(LinReg, (xs,), observations,  10_000, 100, 0.01, LinRegGuide, (), ReinforceELBO());



parameters = get_constrained_parameters(vi_result_2.Q)
mu = vcat(parameters["mu_intercept"], parameters["mu_slope"])
sigma = vcat(parameters["sigma_intercept"], parameters["sigma_slope"])
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)



Random.seed!(0)
vi_result = bbvi(LinReg, (xs,), observations,  10_000, 100, 0.01)
mu = [vi_result.Q[:intercept].base.μ, vi_result.Q[:slope].base.μ]
sigma = [vi_result.Q[:intercept].base.σ, vi_result.Q[:slope].base.σ]
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)

# equivalent to bbvi
ulj = Evaluation.make_unconstrained_logjoint(LinRegStatic, (xs,), observations)
K = length(ulj.addresses_to_ix)
Random.seed!(0)
vi_result_2 = advi_logjoint(ulj.logjoint, 10_000, 100, 0.01, MeanFieldGaussian(K), ReinforceELBO())
maximum(abs, vi_result_2.mu .- mu)
maximum(abs, vi_result_2.sigma .- sigma)



@ppl static function geometric_normal_static()
    n ~ Geometric(0.3)
    #x ~ Normal(n, 0.5)
end

Random.seed!(0)
vi_result = bbvi(geometric_normal_static, (), Dict(), 10_000, 100, 0.01)


@ppl function geometric_normal()
    n ~ Geometric(0.3)
    # x ~ Normal(n, 0.5)
end

Random.seed!(0)
vi_result = bbvi(geometric_normal, (), Dict(),  10_000, 100, 0.01)



import Tracker
Tracker.param([1.]) isa AbstractVector{<:Float64} # true
Tracker.param.([1.]) isa AbstractVector{<:Float64} # false
Tracker.param.([1.]) isa AbstractVector{<:Real} # true


@ppl function unif()
    x ~ Uniform(-1,1)
    y ~ Uniform(x-1,x+1)
    z ~ Uniform(y-1,y+1)
end

Random.seed!(0)
@time vi_result = advi_meanfield(unif, (), Dict(),  10_000, 10, 0.01)
posterior = sample_posterior(vi_result, 1_000_000);
histogram(posterior[:y], normalize=true, legend=false)

Random.seed!(0)
@time vi_result = bbvi(unif, (), Dict(),  10_000, 10, 0.01)
posterior = sample_posterior(vi_result, 1_000_000);
histogram(posterior[:y], normalize=true, legend=false)


import TinyPPL.Distributions: transform_to, TransformedDistribution, RealInterval
@ppl function unif_guide()
    x_mu = param("x_mu")
    x_sigma = param("x_sigma", 1, Positive())
    y_mu = param("y_mu")
    y_sigma = param("y_sigma", 1, Positive())
    z_mu = param("z_mu")
    z_sigma = param("z_sigma", 1, Positive())

    x ~ TransformedDistribution(Normal(x_mu, x_sigma), transform_to(RealInterval(-1,1)))

    y ~ TransformedDistribution(Normal(y_mu, y_sigma), transform_to(RealInterval(x-1,x+1)))

    z ~ TransformedDistribution(Normal(z_mu, z_sigma), transform_to(RealInterval(y-1,y+1)))
end


Random.seed!(0)
@time vi_result = advi(unif, (), Dict(),  10_000, 10, 0.01, unif_guide, (), MonteCarloELBO())
posterior = sample_posterior(vi_result, 1_000_000);
histogram(posterior[:y], normalize=true, legend=false)


@ppl function nunif()
    n ~ Geometric(0.3)
    x = 0.
    for i in 0:Int(n)
        x = {(:x,i)} ~ Uniform(x-1,x+1)
    end
end

Random.seed!(0)
vi_result = bbvi(nunif, (), Dict(),  10_000, 100, 0.01)
vi_result.Q[:n]
theta = sample_posterior(vi_result, 100_000);
theta = transform_to_constrained(zeta, nunif, (), Dict());
histogram(theta[:n], normalize=true, legend=false)
addr = (:x,10)
histogram(theta[addr], normalize=true, legend=false)
# histogram(theta[addr][.!ismissing.(theta[addr])], normalize=true, legend=false)


# TODO: can be used for tests
import Tracker
mu = Tracker.param(2.)
sigma = Tracker.param(0.5)
q = VariationalNormal()
q = Evaluation.update_params(q, [mu, sigma])
x = Tracker.data(rand(q))
# x = rand(q) # does not work
lp = logpdf(q, x)
Tracker.back!(lp)
Tracker.grad(x)
Tracker.grad(q.base.μ)
Tracker.grad(q.log_σ)
Tracker.grad.(Evaluation.get_params(q))

all(Tracker.grad.(Evaluation.get_params(q)) .≈ Evaluation.logpdf_param_grads(q, Tracker.data(x)))


# T = transform_to(support(Uniform(0.,1.)))
# mu = Tracker.param(2.)
# sigma = Tracker.param(0.5)
# q = TransformedVariationalWrappedDistribution(VariationalNormal(), T)
# q = Evaluation.update_params(q, [mu, sigma])
# x = Tracker.data(rand(q))

# lp = logpdf(q, x)
# Tracker.back!(lp)
# Tracker.grad.(Evaluation.get_params(q))
# all(Tracker.grad.(Evaluation.get_params(q)) .≈ Evaluation.logpdf_param_grads(q, Tracker.data(x)))


p = Tracker.param(0.3)
q = VariationalGeometric()
q = Evaluation.update_params(q, [p])
x = 10
lp = logpdf(q, x)
Tracker.back!(lp)
Tracker.grad(x)
Tracker.grad(q.inv_sigmoid_p)
Tracker.grad.(Evaluation.get_params(q))

all(Tracker.grad.(Evaluation.get_params(q)) .≈ Evaluation.logpdf_param_grads(q, Tracker.data(x)))


# PathDerivativeELBO ELBO gradient
import Tracker
using Distributions

f(x, y) = x^2 + 4*x*y
∇f(x,y) = [2x + 4*y, 4x]
g1(a,b) = 3*a + b^3
∇g1(a,b) = [3, 3*b^2]
g2(a,b) = exp(a)*b
∇g2(a,b) = [exp(a)*b, exp(a)]


a = Tracker.param(0.5)
b = Tracker.param(2.0)
r = f(g1(a,b),g2(a,b))
Tracker.back!(r)
Tracker.grad(a)
Tracker.grad(b)

a = Tracker.data(a)
b = Tracker.data(b)
g1_ = g1(a,b)
g2_ = g2(a,b)


DF = reshape(∇f(g1_,g2_), 1, :)
DG = vcat(reshape(∇g1(a,b), 1, :), reshape(∇g2(a,b), 1, :))

DF*DG
transpose(DG) * transpose(DF)

∇ga = [∇g1(a,b)[1], ∇g2(a,b)[1]]
∇gb = [∇g1(a,b)[2], ∇g2(a,b)[2]]

∇f(g1_,g2_)'∇ga
∇f(g1_,g2_)'∇gb

∇f(g1_,g2_)[1] * ∇g1(a,b) + ∇f(g1_,g2_)[2] *  ∇g2(a,b)



p = Normal(2., 0.5)

q = Normal(0., exp(0.))
z = rand(q, 10^7)
elbo = mean(logpdf.(p, z) .- logpdf.(q, z))


μ = Tracker.param(0.)
log_σ = Tracker.param(0.)
q = Normal(μ, exp(log_σ))

Tracker.back!(-kldivergence(q, p))
Tracker.grad(μ)
Tracker.grad(log_σ)

# MonteCarloELBO
μ = Tracker.param(0.)
log_σ = Tracker.param(0.)
q = Normal(μ, exp(log_σ))
z = rand(q, 10^6);
elbo = mean(logpdf.(p, z) .- logpdf.(q, z))
Tracker.back!(elbo)
Tracker.grad(μ)
Tracker.grad(log_σ)

# RelativeEntropyELBO
μ = Tracker.param(0.)
log_σ = Tracker.param(0.)
q = Normal(μ, exp(log_σ))
z = rand(q, 10^6);
elbo = mean(logpdf.(p, z)) + entropy(q)
Tracker.back!(elbo)
Tracker.grad(μ)
Tracker.grad(log_σ)

# ReinforceELBO
μ = Tracker.param(0.)
log_σ = Tracker.param(0.)
q = Normal(μ, exp(log_σ))
z = rand(q, 10^6);
z_ = Tracker.data.(z);
lpq = logpdf.(q, z_);
lpq_ = Tracker.data.(lpq);
elbo_ = logpdf.(p, z_) .- lpq_
elbo = mean(elbo_ .* lpq .+ elbo_ .* (1 .- lpq_))
Tracker.back!(elbo)
Tracker.grad(μ)
Tracker.grad(log_σ)

# PathDerivativeELBO
μ = Tracker.param(0.)
log_σ = Tracker.param(0.)
q = Normal(μ, exp(log_σ))
z = rand(q, 10^6);
q_ = Normal(Tracker.data(q.μ), Tracker.data(q.σ))
elbo = mean(logpdf.(p, z) .- logpdf.(q_, z))
Tracker.back!(elbo)
Tracker.grad(μ)
Tracker.grad(log_σ)



μ = Tracker.param(0.)
log_σ = Tracker.param(0.)
q = Normal(μ, exp(log_σ))
z = rand(q)
Tracker.back!(logpdf(q, z))
Tracker.grad(μ) # = ∇ entropy
Tracker.grad(log_σ) # = ∇ entropy
# ∇ log(1/sqrt(2π σ^2)) - (σ ζ + μ - μ)^2 / σ^2
# d/dω log(1/sqrt(2π exp(ω)^2)) = -1

import PDMats: PDMat
import LinearAlgebra: Cholesky, LowerTriangular, det, Diagonal
μ = Tracker.param.(zeros(3))
L = Tracker.param.([2 0 0; 1 2 0; 0 1 2])
q = MultivariateNormal(μ, PDMat(Cholesky(LowerTriangular(L))))
z = rand(q)
Tracker.back!(logpdf(q, z))
Tracker.grad.(μ) # = ∇ entropy
Tracker.grad.(L) # = ∇ entropy

# ∇ log(1/sqrt(2π det(L*L'))) - (L ζ + μ - μ)' inv(L*L) * (L ζ + μ - μ)
# d/dω log(1/sqrt(2π det(L*L'))) = -1/diag(L)

L = Tracker.param.([2 0 0; 1 2 0; 0 1 2])
LL = LowerTriangular(L)
Tracker.back!( -log(sqrt(2π * det(LL * LL'))))
Tracker.grad.(L) # = ∇ entropy

inv(Diagonal(Tracker.data(L)))
