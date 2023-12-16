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
S = inv(inv(S0) + Phi'Phi / σ^2) 
map_mu = S*(inv(S0) * m0 + Phi'ys / σ^2)

map_Σ = S
map_mu
map_sigma = [sqrt(S[1,1]), sqrt(S[2,2])]

@ppl static function LinRegStatic(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)
    # println("intercept: ", intercept, ", slope: ", slope)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

observations = Dict((:y, i) => y for (i, y) in enumerate(ys));

Random.seed!(0)

addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(LinRegStatic, (xs,), observations);
K = length(addresses_to_ix)

Random.seed!(0)
result = hmc_logjoint(logjoint, K, 10_000, 10, 0.1)
traces = Traces(addresses_to_ix, result)
mean(traces[:slope])
mean(traces[:intercept])

Random.seed!(0)
traces = hmc(LinRegStatic, (xs,), observations, 10_000, 10, 0.1)

plot(traces[:slope]);
plot!(traces[:intercept])

maximum(abs, mean(traces[:intercept]) - map_mu[1])
maximum(abs, mean(traces[:slope]) - map_mu[2])

Random.seed!(0)
mu, sigma = advi_meanfield_logjoint(logjoint, K, 10_000, 10, 0.01)
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)

Random.seed!(0)
Q = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())
# equivalent to advi_meanfield_logjoint
maximum(abs, Q.mu .- mu)
maximum(abs, Q.sigma .- sigma)

Random.seed!(0)
Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO());
# equivalent to MonteCarloELBO because ∇ log Q = ∇ entropy
maximum(abs, Q2.mu .- Q.mu)
maximum(abs, Q2.sigma .- Q.sigma)

Random.seed!(0)
Q3 = advi_logjoint(logjoint, 10_000, 10, 0.01, MixedMeanField([VariationalNormal(), VariationalNormal()]), MonteCarloELBO())
Q3_mu = [d.base.μ for d in Q3.dists]
Q3_sigma = [d.base.σ for d in Q3.dists]
# equivalent to MeanFieldGaussian + MonteCarloELBO()
maximum(abs, Q2.mu .- Q3_mu)
maximum(abs, Q2.sigma .- Q3_sigma)


Random.seed!(0)
Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), ReinforceELBO())
maximum(abs, Q2.mu .- map_mu)
maximum(abs, Q2.sigma .- map_sigma)

maximum(abs, Q2.mu .- Q.mu)
maximum(abs, Q2.sigma .- Q.sigma)

Random.seed!(0)
Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), PathDerivativeELBO())
maximum(abs, Q2.mu .- Q.mu)
maximum(abs, Q2.sigma .- Q.sigma)

maximum(abs, Q2.mu .- Q.mu)
maximum(abs, Q2.sigma .- Q.sigma)

N = 10_000
Random.seed!(0)
mu, L = advi_fullrank_logjoint(logjoint, K, N, 10, 0.01);
maximum(abs, map_mu .- mu)
maximum(abs, map_Σ .- L*L')

Random.seed!(0)
Q = advi_logjoint(logjoint, N, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO());
# equivalent to advi_fullrank_logjoint
maximum(abs, mu .- Q.base.μ)
maximum(abs, L*L' .- Q.base.Σ)

Random.seed!(0)
Q2 = advi_logjoint(logjoint, N, 10, 0.01, FullRankGaussian(K), MonteCarloELBO());
# equivalent to RelativeEntropyELBO because ∇ log Q = ∇ entropy
maximum(abs, Q2.base.μ .- Q.base.μ)
maximum(abs, Q2.base.Σ .- Q.base.Σ)

Random.seed!(0)
Q2 = advi_logjoint(logjoint, N, 100, 0.01, FullRankGaussian(K), ReinforceELBO());
maximum(abs, map_mu .- Q2.base.μ)
maximum(abs, map_Σ .- Q2.base.Σ)

maximum(abs, Q2.base.μ .- Q.base.μ)
maximum(abs, Q2.base.Σ .- Q.base.Σ)

Random.seed!(0)
Q2 = advi_logjoint(logjoint, N, 10, 0.01, FullRankGaussian(K), PathDerivativeELBO());
maximum(abs, map_mu .- Q2.base.μ)
maximum(abs, map_Σ .- Q2.base.Σ)

maximum(abs, Q2.base.μ .- Q.base.μ)
maximum(abs, Q2.base.Σ .- Q.base.Σ)


@ppl static function LinRegGuide()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = exp(param("omega_intercept"))
    sigma2 = exp(param("omega_slope"))

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end

@ppl static function LinRegGuide2()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = param("sigma_intercept", 1, :positive)
    sigma2 = param("sigma_slope", 1, :positive)

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
@time Q = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO())

guide = make_guide(LinRegGuide, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, guide, MonteCarloELBO())
params = get_constrained_parameters(Q2)

mu = vcat(params["mu_intercept"], params["mu_slope"])
sigma = exp.(vcat(params["omega_intercept"], params["omega_slope"]))
# equivalent to advi_logjoint
maximum(abs, mu .- Q.mu)
maximum(abs, sigma .- Q.sigma)


guide = make_guide(LinRegGuide2, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, guide, MonteCarloELBO())
params = get_constrained_parameters(Q2)

mu = vcat(params["mu_intercept"], params["mu_slope"])
sigma = vcat(params["sigma_intercept"], params["sigma_slope"])
# equivalent to advi_logjoint
maximum(abs, mu .- Q.mu)
maximum(abs, sigma .- Q.sigma)


using Plots
@ppl static function unif()
    x ~ Uniform(-1,1)
    y ~ Uniform(x-1,x+1)
    z ~ Uniform(y-1,y+1)
end

Random.seed!(0)
traces, retvals, lp = likelihood_weighting(unif, (), Dict(), 1_000_000);
histogram(traces[:z], weights=exp.(lp), normalize=true, legend=false)

Random.seed!(0)
traces = hmc(unif, (), Dict(), 100_000, 10, 0.1)
histogram(traces[:y], normalize=true, legend=false)


addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(unif, (), Dict());
K = length(addresses_to_ix)

Random.seed!(0)
Q = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())
Q = advi_logjoint(logjoint, 10_000, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO())
zeta = rand(Q, 1_000_000);
theta = transform_to_constrained!(zeta);
histogram(theta[addresses_to_ix[:z],:], normalize=true, legend=false)



@ppl function LinReg(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

observations = Dict((:y, i) => y for (i, y) in enumerate(ys));

Random.seed!(0)
Q = advi_meanfield(LinReg, (xs,), observations,  10_000, 10, 0.01)
mu = [Q[:intercept].base.μ, Q[:slope].base.μ]
sigma = [Q[:intercept].base.σ, Q[:slope].base.σ]
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)

Random.seed!(0)
Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO())
maximum(abs, Q2.mu .- mu)
maximum(abs, Q2.sigma .- sigma)

guide = make_guide(LinRegGuide, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, guide, MonteCarloELBO())
guide_mu = vcat(Q2.sampler.phi[Q2.sampler.params_to_ix["mu_intercept"]], Q2.sampler.phi[Q2.sampler.params_to_ix["mu_slope"]])
guide_sigma = exp.(vcat(Q2.sampler.phi[Q2.sampler.params_to_ix["omega_intercept"]], Q2.sampler.phi[Q2.sampler.params_to_ix["omega_slope"]]))
maximum(abs, guide_mu .- mu)
maximum(abs, guide_sigma .- sigma)

Random.seed!(0)
Q = bbvi(LinReg, (xs,), observations,  10_000, 100, 0.01)
mu = [Q[:intercept].base.μ, Q[:slope].base.μ]
sigma = [Q[:intercept].base.σ, Q[:slope].base.σ]
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)

# equivalent to bbvi
Random.seed!(0)
Q2 = advi_logjoint(logjoint, 10_000, 100, 0.01, MeanFieldGaussian(K), ReinforceELBO())
maximum(abs, Q2.mu .- mu)
maximum(abs, Q2.sigma .- sigma)



@ppl static function geometric_normal_static()
    n ~ Geometric(0.3)
    #x ~ Normal(n, 0.5)
end

addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(geometric_normal_static, (), Dict());

Q = get_mixed_meanfield(geometric_normal_static, (), Dict(), addresses_to_ix);
Random.seed!(0)
Q = advi_logjoint(logjoint, 10_000, 100, 0.01, Q, ReinforceELBO())


@ppl function geometric_normal()
    n ~ Geometric(0.3)
    x ~ Normal(n, 0.5)
end

Random.seed!(0)
Q = bbvi(geometric_normal, (), Dict(),  10_000, 100, 0.01)

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
@time Q = advi_meanfield(unif, (), Dict(),  10_000, 10, 0.01)
zeta = rand(Q, 1_000_000);
theta = universal_transform_to_constrained(zeta, unif, (), Dict());
histogram([t[:y] for t in theta], normalize=true, legend=false)

Random.seed!(0)
@time Q = bbvi(unif, (), Dict(),  10_000, 10, 0.01)
zeta = rand(Q, 1_000_000);
theta = universal_transform_to_constrained(zeta, unif, (), Dict());
histogram([t[:y] for t in theta], normalize=true, legend=false)

@ppl function nunif()
    n ~ Geometric(0.3)
    x = 0.
    for i in 0:Int(n)
        x = {(:x,i)} ~ Uniform(x-1,x+1)
    end
end

Random.seed!(0)
Q = bbvi(nunif, (), Dict(),  10_000, 100, 0.01)
Q[:n]
zeta = rand(Q, 100_000);
theta = universal_transform_to_constrained(zeta, nunif, (), Dict());
histogram([t[(:n)] for t in theta], normalize=true, legend=false)
addr = (:x,1)
histogram([t[addr] for t in theta if haskey(t,addr)], normalize=true, legend=false)


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


@ppl function LinReg(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end

    return (slope, intercept)
end

observations = Dict((:y, i) => y for (i, y) in enumerate(ys));

N = 1_000_000

Random.seed!(0)
traces, retvals, logprobs = rwmh(LinReg, (xs,), observations, N;
    default_var=1., addr2var=Addr2Var(:intercept=>0.5,:slope=>0.3), gibbs=false);

mean(t[:slope] for t in traces)
mean(t[:intercept] for t in traces)

Random.seed!(0)
traces, retvals, logprobs = lmh(LinReg, (xs,), observations, N;
    proposal=Proposal(
        :intercept => ContinuousRandomWalkProposal(0.5),
        :slope => ContinuousRandomWalkProposal(0.3),
    ), gibbs=false);

mean(t[:slope] for t in traces)
mean(t[:intercept] for t in traces)


@ppl static function static_normal()
    x ~ Normal(2., 0.5)
end
observations = Dict()

addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(static_normal, (), observations);
K = length(addresses_to_ix)

Random.seed!(0)
mu, sigma = advi_meanfield_logjoint(logjoint, K, 10_000, 10, 0.01)


#Random.seed!(0)
#mu, sigma = advi_meanfield(logjoint, 100_000, 100, 0.001, K)

Random.seed!(0)
Q = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())

Random.seed!(0)
Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, MeanFieldGaussian(K), MonteCarloELBO())

@ppl static function static_normal_guide()
    mu = param("mu")
    sigma = param("sigma", 1, :positive)
    x ~ Normal(mu, sigma)
end

guide = make_guide(static_normal_guide, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time Q2 = advi_logjoint(logjoint, 10_000, 10, 0.01, guide, MonteCarloELBO())
params = get_constrained_parameters(Q2)


@ppl function universal_normal()
    x ~ Normal(2., 0.5)
end
observations = Dict()

Random.seed!(0)
Q = advi_meanfield(universal_normal, (), observations,  10_000, 10, 0.01)
Q[:x].base

Random.seed!(0)
Q = bbvi(universal_normal, (), observations,  10_000, 10, 0.01)
Q[:x].base



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
# d/dω log(1/sqrt(2π exp(ω)^2)) = -1

L = Tracker.param.([2 0 0; 1 2 0; 0 1 2])
LL = LowerTriangular(L)
Tracker.back!( -log(sqrt(2π * det(LL * LL'))))
Tracker.grad.(L) # = ∇ entropy

inv(Diagonal(Tracker.data(L)))
