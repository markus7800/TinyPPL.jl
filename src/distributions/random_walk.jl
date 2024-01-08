import Distributions: Distribution, ContinuousUnivariateDistribution, DiscreteUnivariateDistribution, ncategories, truncated

# function continuous_interval(lower::Real, upper::Real, x::Real, var::Real)
#     width = upper - lower
#     # rescale
#     μ = clamp((x - lower) / width, 1e-3, 1-1e-3) # ∈ [0, 1]
#     σ2 = clamp(var * 1. / width^2, 1e-3, μ * (1-μ) - 1e-3)

#     # @assert σ^2 < μ * (1-μ) (var < width/2 necessary)

#     α = ((1.0 - μ) / σ2 - 1/μ) * μ^2
#     β = α * (1/μ - 1.)

#     return width * Beta(α, β) + lower
# end

# function continuous_greater_than(lower::Real, x::Real, var::Real)
#     μ = x - lower # ∈ [0, ∞)
#     α = max(μ^2 / var, 1e-3)
#     θ = max(μ / α, 1e-3)
#     return Gamma(α, θ) + lower
# end

# supports lower = -Inf or upper = Inf
struct ContinuousRWProposer <: ContinuousUnivariateDistribution
    a::Real
    l::Distribution
    b::Real
    u::Distribution
    μ::Float64
    bp::Float64
    function ContinuousRWProposer(lower::Real, upper::Real, x::Float64, var::Real)
        var = var / 2
        l = truncated(Normal(x, sqrt(var)), lower, max(x, lower+1e-3))
        u = truncated(Normal(x, sqrt(var)), min(x, upper-1e-3), upper)

        mass_left = l.tp
        mass_right = u.tp
        bp = mass_right / (mass_left + mass_right)
        return new(lower, l, upper, u, x, bp)
    end
end

function Base.rand(d::ContinuousRWProposer)::Float64
    if rand() < d.bp
        return rand(d.u)
    else
        return rand(d.l)
    end
end

function logpdf(d::ContinuousRWProposer, x::Float64)::Float64
    if d.a <= x <= d.μ
        return logpdf(d.l, x) + log(1-d.bp)
    elseif d.μ <= x <= d.b
        return logpdf(d.u, x) + log(d.bp)
    end
    return -Inf
end

function continuous_interval(lower::Real, upper::Real, x::Real, var::Real)
    return ContinuousRWProposer(lower, upper, x, var)
end

function continuous_greater_than(lower::Real, x::Real, var::Real)
    return ContinuousRWProposer(lower, Inf, x, var)
end

# supports upper = Inf
struct DiscreteRWProposer <: DiscreteUnivariateDistribution
    a::Int
    la::Float64
    b::Real
    lb::Float64
    μ::Int
    G::Distribution
    bp::Float64
    function DiscreteRWProposer(lower::Int, upper::Real, x::Int, var::Real)
        a = x - lower
        b = upper - x
        var = var / 2
        p = (sqrt(1 + 4*var) - 1) / (2*var)
        G = Geometric(p)
        if a != 0 && b != 0
            mass_left = 1-(1-p)^a
            mass_right = 1-(1-p)^b
            bp = mass_right / (mass_left + mass_right)
            la = log((1-bp) / mass_left)
            lb = log(bp / mass_right)
        else
            bp = NaN
            la = -log(1-(1-p)^a)
            lb = -log(1-(1-p)^b)
        end
        return new(a, la, b, lb, x, G, bp)
    end
end

function Base.rand(d::DiscreteRWProposer)::Int
    if d.b == 0
        return d.μ - rand(d.G) % d.a - 1
    end
    if d.a == 0
        return rand(d.G) % d.b + d.μ + 1
    end

    if rand() < d.bp
        return rand(d.G) % d.b + d.μ + 1
    else
        return d.μ - rand(d.G) % d.a - 1
    end
end

function logpdf(d::DiscreteRWProposer, x::Real)::Float64
    x = Int(x)
    y = x - d.μ
    if 0 < y <= d.b
        return logpdf(d.G, y-1) + d.lb
    elseif -d.a <= y < 0
        return logpdf(d.G, -y-1) + d.la
    end
    return -Inf
end

function discrete_interval(lower::Real, upper::Real, x::Real, var::Real)
    return DiscreteRWProposer(Int(lower), Int(upper), Int(x), var)
end

function discrete_greater_than(lower::Real, x::Real, var::Real)
    return DiscreteRWProposer(Int(lower), Inf, Int(x), var)
end

export continuous_interval, continuous_greater_than, discrete_interval, discrete_greater_than

import Distributions: Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
import Distributions: Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform

random_walk_proposal_dist(d::Bernoulli, value::Real, var::Real) = value > 0 ? Bernoulli(0.) : Bernoulli(1.)
random_walk_proposal_dist(d::Binomial, value::Real, var::Real) = discrete_interval(0, d.n, value, var)
random_walk_proposal_dist(d::Categorical, value::Real, var::Real) = discrete_interval(1, ncategories(d), value, var)
random_walk_proposal_dist(d::DiscreteUniform, value::Real, var::Real) = discrete_interval(d.a, d.b, value, var)
random_walk_proposal_dist(d::Geometric, value::Real, var::Real) = discrete_greater_than(0, value, var)
random_walk_proposal_dist(d::Poisson, value::Real, var::Real) = discrete_greater_than(0, value, var)

random_walk_proposal_dist(d::Beta, value::Real, var::Real) = continuous_interval(0., 1., value, var)
random_walk_proposal_dist(d::Cauchy, value::Real, var::Real) = Normal(value, sqrt(var))
random_walk_proposal_dist(d::Exponential, value::Real, var::Real) = continuous_greater_than(0, value, var)
random_walk_proposal_dist(d::Gamma, value::Real, var::Real) = continuous_greater_than(0, value, var)
random_walk_proposal_dist(d::InverseGamma, value::Real, var::Real) = continuous_greater_than(0, value, var)
random_walk_proposal_dist(d::Laplace, value::Real, var::Real) = Normal(value, sqrt(var))
random_walk_proposal_dist(d::LogNormal, value::Real, var::Real) = continuous_greater_than(0, value, var)
random_walk_proposal_dist(d::Normal, value::Real, var::Real) = Normal(value, sqrt(var))
random_walk_proposal_dist(d::TDist, value::Real, var::Real) = Normal(value, sqrt(var))
random_walk_proposal_dist(d::Uniform, value::Real, var::Real) = continuous_interval(d.a, d.b, value, var)

const rw_proposal_dist = random_walk_proposal_dist

export random_walk_proposal_dist, rw_proposal_dist, ContinuousRWProposer, DiscreteRWProposer, ncategories

# mapping of address to variance

const Addr2Var = MostSpecificDict{Float64}
function Addr2Var(args...)
    return MostSpecificDict(Dict{Any, Float64}(args...))
end
export Addr2Var

# using Plots
# d = continuous_interval(-1, 3, 1.5, 0.5)
# Distributions.mean(d)
# Distributions.var(d)
# plot(x -> exp(Distributions.logpdf(d, x)), xlims=(-1,3))

# import Distributions: mean, var

# d = continuous_greater_than(-1, 1.5, 0.5)
# mean(d)
# var(d)
# plot(x -> exp(Distributions.logpdf(d, x)), xlims=(-1,3))


# d = discrete_interval(1, 6, 3, .5)
# ps = exp.(Distributions.logpdf.(d, 0:7))
# bar(0:7, ps)

# sum(ps[1:3])
# sum(ps[4:7])

# xs = [rand(d) for i in 1:10^7]
# [Distributions.mean(xs .== i) for i in 1:6]


# d = discrete_greater_than(-2, 3, 5.)
# ps = exp.(Distributions.logpdf.(d, -3:15))
# bar(-3:15, ps)
# sum(ps)

# using TinyPPL.Distributions
# d = random_walk_proposal_dist(Categorical([0.25, 0.25, 0.25, 0.25]), 2, 1.)
# ps = exp.(logpdf.(d, 0:5))
# X = [rand(d) for _ in 1:10^7]
# ps_hat = [mean(X .== i) for i in 1:5]