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
Q = advi(logjoint, 10_000, 10, 0.01, Q, RelativeEntropyELBO())



N = 10_000
Random.seed!(0)
mu, L = advi_fullrank(logjoint, N, 10, 0.01, K);
Random.seed!(0)
Q = advi(logjoint, N, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO());
Random.seed!(0)
Q2 = advi(logjoint, N, 10, 0.01, FullRankGaussian(K), MonteCarloELBO());

maximum(abs.(mu .- Q.base.μ))
maximum(abs.(L*L' .- Q.base.Σ))

maximum(abs.(Q2.base.μ .- Q.base.μ))
maximum(abs.(Q2.base.Σ .- Q.base.Σ))


Random.seed!(0)
mu, L = advi_fullrank(logjoint, 10_000, 100, 0.01, K)

L*L'

Q = FullRankGaussian(K)
Random.seed!(0)
Q = advi(logjoint, 10_000, 10, 0.01, Q, RelativeEntropyELBO())
Q = advi(logjoint, 10_000, 10, 0.01, Q, MonteCarloELBO())

@ppl static function LinRegGuide()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = exp(param("sigma_intercept"))
    sigma2 = exp(param("sigma_slope"))

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end

@ppl static function FullRankLinRegGuide()
    mu = param("mu", 2)
    L = param("L", 4)
    zeta = {:zeta} ~ MvNormal(mu, L*L') # TODO

    {:intercept} ~ Dirac(zeta[1])
    {:slope} ~ Dirac(zeta[2])
end

Q = MeanFieldGaussian(K)
phi = Tracker.param(vcat(zeros(K), ones(K)))
phi = Tracker.param(vcat(ones(K), ones(K)))
Q = Evaluation.update_params(Q, phi)

zeta, lpq = Evaluation.rand_and_logpdf(Q)
Tracker.back!(lpq)
phi.grad

Random.seed!(0)
elbo = Evaluation.estimate_elbo(RelativeEntropyELBO(), logjoint, Q, 1)
Tracker.back!(elbo)
phi.grad

Random.seed!(0)
elbo = Evaluation.estimate_elbo(MonteCarloELBO(), logjoint, Q, 1)
Tracker.back!(elbo)
phi.grad

Random.seed!(0);
@time Q = advi(logjoint, 10_000, 100, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())

Random.seed!(0);
@time Q = advi(logjoint, 10_000, 100, 0.01, MeanFieldGaussian(K), MonteCarloELBO())

guide = make_guide(LinRegGuide, (), Dict(), addresses_to_ix)
Random.seed!(0)
@time Q = advi(logjoint, 10_000, 100, 0.01, guide, MonteCarloELBO())

x = Tracker.param.([1.,1.])
vcat(x)

eltype(x)
zeros(eltype(x), size(x))

x[1] = x[2]

using Distributions
import Tracker
import LinearAlgebra
import PDMats
import Random

Random.seed!(0)
K = 3
params = randn(K + K^2)
params = Tracker.param(params)

mu = params[1:K]
A = reshape(params[K+1:end], K, K)
A = convert(Matrix{eltype(A)}, A) # Tracked K×K Matrix{Float64} -> K×K Matrix{Tracker.TrackedReal{Float64}}

L = LinearAlgebra.LowerTriangular(A)
d = Distributions.MultivariateNormal(mu, PDMats.PDMat(LinearAlgebra.Cholesky(L)))

Random.seed!(0)
x = rand(d)
Tracker.back!(sum(x))
params.grad

inv(d.Σ.chol.L)*(x .- d.μ)


Random.seed!(0)
eta = randn(K)
x = d.Σ.chol.L*eta + d.μ
Tracker.back!(sum(x))
params.grad

Random.seed!(0)
var(rand(d, 10^6), dims=2)

d2 = MultivariateNormal(mu, L*L')
Random.seed!(0)
var(rand(d, 10^6), dims=2)




import Distributions
import Tracker
function Distributions._rand!(rng::Distributions.AbstractRNG, d::Distributions.MvNormal, x::VecOrMat)
#     [2] _rand!(rng::Random.TaskLocalRNG, d::Distributions.MvNormal{Tracker.TrackedReal{Float64}, PDMats.PDMat{Tracker.TrackedReal{Float64}, Matrix{Tracker.TrackedReal{Float64}}}, Vector{Tracker.TrackedReal{Float64}}}, x::Vector{Tracker.TrackedReal{Float64}})
#     @ Main ~/Documents/TinyPPL.jl/examples/advi/evaluation.jl:169
#     [3] rand!
#     @ ~/.julia/packages/Distributions/Ufrz2/src/genericrand.jl:91 [inlined]
#     [4] rand(rng::Random.TaskLocalRNG, s::Distributions.MvNormal{Tracker.TrackedReal{Float64}, PDMats.PDMat{Tracker.TrackedReal{Float64}, Matrix{Tracker.TrackedReal{Float64}}}, Vector{Tracker.TrackedReal{Float64}}})
#     @ Distributions ~/.julia/packages/Distributions/Ufrz2/src/genericrand.jl:48
#     [5] rand(::Distributions.MvNormal{Tracker.TrackedReal{Float64}, PDMats.PDMat{Tracker.TrackedReal{Float64}, Matrix{Tracker.TrackedReal{Float64}}}, Vector{Tracker.TrackedReal{Float64}}})
#     @ Distributions ~/.julia/packages/Distributions/Ufrz2/src/genericrand.jl:22
    
    println(x) # Tracker.TrackedReal{Float64}[#undef, #undef]
    error("Here")
    return x
end

d = Distributions.MvNormal(Tracker.param.([1.,1.]), Tracker.param.([1. 0.; 0. 1.]))
# Distributions.MvNormal{Tracker.TrackedReal{Float64}, PDMats.PDMat{Tracker.TrackedReal{Float64}, Matrix{Tracker.TrackedReal{Float64}}}, Vector{Tracker.TrackedReal{Float64}}}(
# dim: 2
# μ: Tracker.TrackedReal{Float64}[1.0, 1.0]
# Σ: Tracker.TrackedReal{Float64}[1.0 0.0; 0.0 1.0]
# )
rand(d)


import Tracker
import Distributions
import Random
import LinearAlgebra
import PDMats
d = Distributions.MvNormal(Tracker.param.([1.,1.]), Tracker.param.([1. 0.; 0. 1.]))
Tracker.back!(sum(rand(d)))
Tracker.grad.(d.μ)

d = Distributions.MvNormal(Tracker.param([1.,1.]), Tracker.param([1. 0.; 0. 1.]))
Tracker.back!(sum(rand(d)))
Tracker.grad.(d.μ)
Tracker.grad.(d.Σ.chol.L)

mu = [1., 2.]
L = [2. 0.; 1. 2.]

mu = Tracker.param.(mu)
L = Tracker.param.(L)

Random.seed!(0);
L * randn(2) + mu
# 2.8859410668922383
# 3.2108160487524877

Random.seed!(0);
d = Distributions.MvNormal(mu, PDMats.PDMat(LinearAlgebra.Cholesky(LinearAlgebra.LowerTriangular(L))));
rand(d)
# produces not the same tracked vs untracked

Random.seed!(0);
d = Distributions.MvNormal(mu, L*L');
rand(d)
# produces also not the same tracked vs untracked


Random.seed!(0);
d = Distributions.MvNormal(mu, PDMats.PDMat(LinearAlgebra.Cholesky(LinearAlgebra.LowerTriangular(L))));
X = rand(d, 10^6)
Distributions.mean(X, dims=2)
Distributions.cov(X, dims=2)

x = Array{Tracker.TrackedReal{Float64}}(undef, 10^6)
Random.randn!(Random.default_rng(), x)
Distributions.mean(x)
Distributions.std(x)

function myrandn!(rng::Random.AbstractRNG, A::AbstractArray{T}) where T
    for i in eachindex(A)
        #@inbounds A[i] = Random.randn(rng, T)
        @inbounds A[i] = convert(T, Random.randn(rng))
    end
    A
end

x = Array{Tracker.TrackedReal{Float64}}(undef, 10^6)
myrandn!(Random.default_rng(), x)
Distributions.mean(x)
Distributions.std(x)

y = Tracker.data.(x)
Distributions.mean(Random.randn!(Random.default_rng(), y))

@which Random.randn(Random.default_rng(), Tracker.TrackedReal{Float64})


for f in :[rand, randn, randexp].args
    @eval Random.$f(rng::Random.AbstractRNG,::Type{Tracker.TrackedReal{T}}) where {T} = Tracker.param(Random.$f(rng,T))
end


#x = [convert(Tracker.TrackedReal{Float64}, Random.randn(Random.default_rng())) for i in 1:10^6]

x = [Random.randn(Random.default_rng(), Tracker.TrackedReal{Float64}) for i in 1:10^6]
Distributions.mean(x)
Distributions.std(x)

using Statistics: mean, std, var
using Random
using Test

@testset "random" begin
    Random.seed!(1234)
    n_samples = 10^6
    x = [Random.rand(Tracker.TrackedReal{Float64}) for i in 1:n_samples]
    @test isapprox(mean(x), 0.5, atol=1e-2, rtol=0)
    @test isapprox(var(x), 1/12, atol=1e-2, rtol=0)

    x = [Random.randn(Tracker.TrackedReal{Float64}) for i in 1:n_samples]
    @test isapprox(mean(x), 0., atol=1e-2, rtol=0)
    @test isapprox(var(x), 1., atol=1e-2, rtol=0)

    x = [Random.randexp(Tracker.TrackedReal{Float64}) for i in 1:n_samples]
    @test isapprox(mean(x), 1., atol=1e-2, rtol=0)
    @test isapprox(var(x), 1., atol=1e-2, rtol=0)
end

mean(x)