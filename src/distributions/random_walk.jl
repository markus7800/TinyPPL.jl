import Distributions: Distribution, ContinuousUnivariateDistribution, DiscreteUnivariateDistribution, ncategories, truncated

# Random walk proposal on the interval [lower, upper] (lower = -Inf or upper = Inf allowed).
# Normal(x, sqrt(var)) truncated to the support.
# (The previous split-normal construction, which picks the left or right half with probability
#  proportional to its mass, is the same distribution.)
function ContinuousRWProposer(lower::Real, upper::Real, x::Real, var::Real)
    return truncated(Normal(x, sqrt(var)), lower, upper)
end

function continuous_interval(lower::Real, upper::Real, x::Real, var::Real)
    return ContinuousRWProposer(lower, upper, x, var)
end

function continuous_greater_than(lower::Real, x::Real, var::Real)
    return ContinuousRWProposer(lower, Inf, x, var)
end

# supports upper = Inf (then b = Inf::Float64, otherwise b::Int)
struct DiscreteRWProposer{B<:Real} <: DiscreteUnivariateDistribution
    a::Int
    la::Float64
    b::B
    lb::Float64
    μ::Int
    G::Geometric{Float64}
    bp::Float64
    function DiscreteRWProposer(lower::Int, upper::B, x::Int, var::Real) where {B<:Real}
        a = x - lower
        b = upper - x
        var = var / 2
        p = (sqrt(1 + 4*var) - 1) / (2*var)
        G = Geometric(p)
        if a == 0 && b == 0
            # single value in support, proposal is point mass at x
            bp = NaN
            la = -Inf
            lb = -Inf
        elseif a != 0 && b != 0
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
        return new{typeof(b)}(a, la, b, lb, x, G, bp)
    end
end

function Base.rand(d::DiscreteRWProposer)::Int
    if d.a == 0 && d.b == 0
        return d.μ
    end
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
    if d.a == 0 && d.b == 0
        return y == 0 ? 0. : -Inf
    end
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

import TinyPPL: MostSpecificDict

const Addr2Var = MostSpecificDict{Float64}
function Addr2Var(args...)
    return MostSpecificDict(Dict{Any, Float64}(args...))
end
export Addr2Var
