

using TinyPPL.Distributions
using TinyPPL.Evaluation

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


@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000, Proposal());
@time traces, retvals, lps = lmh(geometric, (0.5, true), observations, 1_000_000, Proposal(:b=>Bernoulli(0.3)));
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