

using TinyPPL.Distributions
using TinyPPL.Evaluation

import Random

@ppl function urn(K::Int)
    N ~ Poisson(6)
    balls = []
    for i in 1:N
        ball = {:ball => i} ~ Bernoulli(0.5)
        push!(balls, ball)
    end
    n_black = 0
    if N > 0
        for k in 1:K
            ball_ix = {:drawn_ball => k} ~ DiscreteUniform(1,N)
            n_black += balls[ball_ix]
        end
    end
    {:n_black} ~ Dirac(n_black)
    return N
end

observations = Dict(:n_black => 5);

Random.seed!(0)
@time traces, retvals, lps = likelihood_weighting(urn, (10,), observations, 5_000_000); #  12.550633 seconds (185.56 M allocations: 13.448 GiB, 39.63% gc time, 0.37% compilation time)
@time result, retvals, lps = likelihood_weighting(urn, (10,), observations, 5_000_000, Evaluation.no_op_completion); # 5.348895 seconds (95.62 M allocations: 3.476 GiB, 25.26% gc time)
Ns = retvals
W = exp.(lps);
[sum(W[Ns .== n]) for n in 1:15]


@ppl static function normal(N)
    X = {:X} ~ Normal(0., 1.)
    for i in 1:N
        X = {:X => i} ~ Normal(0., 1.)
    end
    Z ~ Normal(X, 1.)
    return X
end
observations = Dict(:Z => 1.);
Random.seed!(0)
@time traces, retvals, lps = likelihood_weighting(normal, (100,), observations, 1_000_000); # 37.068228 seconds (1.01 G allocations: 19.085 GiB, 33.81% gc time, 0.23% compilation time)
@time traces, retvals, lps = likelihood_weighting(normal, (100,), observations, 1_000_000, Set(Any[:X]));

@ppl static function normal()
    X ~ Normal(0., 1.)
    Y ~ Normal(X, 1.)
    Z ~ Normal(Y, 1.)
    return X
end


@ppl function unif()
    X ~ Uniform(-1., 1.)
    Y ~ Uniform(-1. + X, 1. + X)
    Z ~ Uniform(-1. + Y, 1. + Y)
    return X
end

model = normal
model = unif

observations = Dict(:Z => 1.);

addresses = Evaluation.get_addresses(model, (), observations)

Random.seed!(0)
@time traces, retvals, lps = likelihood_weighting(model, (), observations, 1_000_000);
@time traces, retvals, lps = likelihood_weighting(model, (), observations, 5_000_000, Set(Any[:X]));
W = exp.(lps);
W'retvals
((retvals .- W'retvals).^2)'W


using StatsPlots
histogram(retvals, weights=W, normalize=true);
plot!(x -> exp(logpdf(Normal(1/3,sqrt(2/3)), x)))

@time traces, retvals, lps = lmh(model, (), observations, 1_000_000; proposal=Proposal(), gibbs=false);
@time traces, retvals, lps = rwmh(model, (), observations, 1_000_000);

histogram(retvals, normalize=true)
plot!(x -> exp(logpdf(Normal(1/3,sqrt(2/3)), x)))

@ppl function pedestrian(i_max)
    start_position ~ Uniform(0.,3.)
    position = start_position
    distance = 0.
    i = 1
    while position >= 0. && distance <= 10. && i <= i_max
        step = {:step=>i} ~ Uniform(-1.,1.)
        distance += abs(step)
        position += step
        i += 1
    end
    {:distance} ~ Normal(distance, 0.1)

    return start_position
end

observations = Dict(:distance => 1.1);

import Random
retvals, lps = begin
    Random.seed!(0)
    N = 10^8
    lps = Vector{Float64}(undef, N)
    retvals = Vector{Float64}(undef, N)
    sampler = LogProb()
    for i in 1:N
        sampler.log_p_Y = 0.
        sampler.log_p_X = 0.
        retvals[i] = pedestrian(Inf, sampler, observations)
        lps[i] = sampler.log_p_Y
    end
    m = maximum(lps)
    l = m + log(sum(exp, lps .- m))
    lps =  lps .- l

    return retvals, lps
end;
W = exp.(lps);
histogram(retvals, weights=W, normalize=true)
density(retvals, weights=W)

@time traces, retvals, lps = lmh(pedestrian, (Inf,), observations, 1_000_000; proposal=Proposal(), gibbs=false);
@time traces, retvals, lps = rwmh(pedestrian, (Inf,), observations, 1_000_000, default_var=0.1);

histogram(retvals, normalize=true)


d = random_walk_proposal_dist(Uniform(-1,1), -0.5, 1.)
histogram([rand(d) for _ in 1:10^6], normalize=true)
plot!(x -> exp(logpdf(d, x)), xlims=(-1.1,1.1))



@ppl function geometric(p::Float64, observed::Bool)
    i = 0
    while true
        b = {:b => i} ~ Bernoulli(p)
        b && break
        i += 1
    end
    if observed
        {:X} ~ Normal(i, 1.0)
    end
    return i
end

observations = Dict(:X => 5);
sampler = Forward();
@time [geometric(0.5, true, sampler, observations) for _ in 1:5_000_000];

@time traces, retvals, lps = likelihood_weighting(geometric, (0.5, true), observations, 1_000_000);

W = exp.(lps);
P_hat = [sum(W[retvals .== i]) for i in 0:10]


@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000; proposal=Proposal(), gibbs=false);
@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000, proposal=Proposal(:b=>Bernoulli(0.3)), gibbs=false);
@time traces, retvals, lps = rwmh(geometric, (0.5, true), observations, 1_000_000);

P_hat = [mean(retvals .== i) for i in 0:10]

function f(slope, intercept, x)
    intercept + slope * x
end

@ppl function LinReg(xs)
    slope = {:slope} ~ Normal(0.0, 10.)
    intercept ~ Normal(0.0, 10.)

    for i in 1:length(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), 1.)
    end

    return (slope, intercept)
end

xs = [1., 2., 3., 4., 5.]
ys = [2.1, 3.9, 5.3, 7.7, 10.2]

observations = Dict((:y,i) => ys[i] for i in 1:length(ys));
@time traces, retvals, lps = likelihood_weighting(LinReg, (xs,), observations, 1_000_000);
W = exp.(lps);
[r[1] for r in retvals]'W
[r[2] for r in retvals]'W

@time traces, retvals, lps = lmh(LinReg, (xs,), observations, 1_000_000);
@time traces, retvals, lps = rwmh(LinReg, (xs,), observations, 1_000_000, default_var=0.01);
mean(r[1] for r in retvals)
mean(r[2] for r in retvals)


using MacroTools

# MacroTools.@capture(:(x ~ Normal(0, 1)), var_ ~ dist_)

MacroTools.@capture(:({:x} ~ Normal(0, 1)), var_ ~ dist_)


using Tracker
using Distributions

function test(a::Real, b::Real)
    return a * b
end

a = Tracker.param(2.)
b = Tracker.param(3.)
a isa Real
a isa AbstractFloat
c = test(a,b)

x = Tracker.param([1., 2.])
x isa AbstractArray{Float64}

x = Tracker.param([1, 2])
x isa AbstractArray{Float64}

Normal(a, b)



using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("../examples/univariate_gmm/data.jl")

promote_type(TrackedValue{Int64}, Int64)
TrackedValue{Int64} <: TrackedValue{T} where T <: Real

a = TrackedValue(1, Set(Any[1]))
convert(TrackedValue{Float64}, a)
convert(TrackedValue{Float64}, 3)
b = 2
c = TrackedValue(3, Set(Any[1]))
promote(a, b)

promote_type(typeof(a), typeof(b))

T = TrackedValue{Int64}
S = Int64
Base.promote_result(T, S, promote_rule(T,S), promote_rule(S,T))
promote_type(T, S)

abstract type A end
abstract type B end

Base.promote_rule(::Type{A}, ::Type{B}) = B

Base.promote_rule(A, B)
Base.promote_rule(B, A)
Base.promote_type(B, A)

const λ = 3
const δ = 5.0
const ξ = 0.0
const κ = 0.01
const α = 2.0
const β = 10.0

@ppl function dirichlet(δ, k)
    w = [{:w=>j} ~ Gamma(δ, 1) for j in 1:k]
    return w / sum(w)
end

@ppl function gmm()
    k = {:k} ~ Poisson(λ)
    k = k + 1
    w = @subppl dirichlet(δ, k)

    means, vars = zeros(Real, k), zeros(Real, k)
    for j=1:k
        means[j] = ({:μ=>j} ~ Normal(ξ, 1/sqrt(κ)))
        vars[j] = ({:σ²=>j} ~ InverseGamma(α, β))
    end
    for i=1:length(gt_ys)
        z = {:z=>i} ~ Categorical(w)
        z > k && continue # then trace has logpdf -Inf anyways
        {:y=>i} ~ Normal(means[z], sqrt(vars[z]))
    end
end

const observations = Dict{Any, Real}((:y=>i)=>y for (i, y) in enumerate(gt_ys))
observations[:k] = gt_k-1

sampler = DependencyAnalyzer()
gmm(sampler, observations)

for (x, children) in sampler.dependencies
    println(x, ": ", children)
end

isprobvec(p::AbstractVector{<:Real}) = all(x -> x ≥ zero(x), p) && isapprox(sum(p), one(eltype(p)))

p = rand(4)
p /= sum(p)
isprobvec(p)

q = TrackedValue.(p)
isprobvec(q)

isapprox(sum(q), one(eltype(q)))

for i in 1:(TrackedValue(10))
    println(i)
end

using TinyPPL.Evaluation

r = 1:(TrackedValue(10))

start, stop = first(r), last(r)
10 == stop

a = oneunit(zero(stop) - zero(start))
Integer(a)
length(r)
collect(r)

n::Int = length(r)
a = Vector{TrackedValue{Int64}}(undef, n)
i = 1
for x in r
    println(i, ": ", x)
    a[i] = x
    i += 1
end
return a

import ProgressLogging: @progress
@profview for _ in 1:10^5
    gmm(Forward(), observations) 
end


using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

@ppl static function normal()
    X ~ Normal(0., 1.)
    Y ~ Normal(X, 1.)
    Z ~ Normal(Y, 1.)
    return X
end
observations = Dict{Any,Real}(:Z => 1.)

addresses = Evaluation.get_addresses(normal, (), observations)
addresses_to_ix = Evaluation.get_address_to_ix(addresses)
sampler = Evaluation.LogJoint(addresses_to_ix, Float64[1.,1.])

@code_warntype sample(sampler, :X, Normal(0.,1), nothing)

@time sample(sampler, :X, Normal(1.,1), nothing)

lj = Evaluation.make_logjoint(normal, (), observations)
@time lj(Float64[1.,1.])



@ppl static function unif()
    X ~ Uniform(-1., 1.)
    Y ~ Uniform(-1. + X, 1. + X)
    Z ~ Uniform(-1. + Y, 1. + Y)
    return X
end
observations = Dict{Any,Real}(:Z => 1.)
addresses_to_ix, logjoint, transform_to_constrained, transform_to_unconstrained = Evaluation.make_unconstrained_logjoint(unif, (), observations)
transform_to_unconstrained([0.5, 1.0])
transform_to_constrained([0.5,100.])
sampler = Evaluation.UnconstrainedLogJoint(addresses_to_ix, [0.5, 1.0])
import Tracker
x = Tracker.param([0.5, 1.0])
sampler = Evaluation.UnconstrainedLogJoint(addresses_to_ix, x)
typeof(sampler.W)

@ppl static function unif()
    X ~ Uniform(-1., 1.)
    Y ~ Uniform(-1. + X, 1. + X)
    Z ~ Uniform(-1. + Y, 1. + Y)
    return X
end
observations = Dict{Any,Real}(:Z => 1.)
observations = Dict{Any,Real}()
addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained! = Evaluation.make_unconstrained_logjoint(unif, (), observations)

K = length(addresses_to_ix)
Random.seed!(0)
mu, sigma = advi(logjoint, 10_000, 10, 0.01, K)

Q = MeanFieldGaussian(K)
Random.seed!(0)
Q = advi(logjoint, 10_000, 10, 0.01, Q)

using Plots

posterior = sigma .* randn(K, 1_000_000) .+ mu
constrained_posterior = transform_to_constrained!(copy(posterior));
histogram(constrained_posterior[addresses_to_ix[:Y],:], normalize=true)


function test(x::T, xs::V) where {T <: Real, V <: AbstractArray{T}}
    return x, xs
end

