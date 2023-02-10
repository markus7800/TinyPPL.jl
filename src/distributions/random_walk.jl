import Distributions: DiscreteUnivariateDistribution

function continuous_interval(lower::Real, upper::Real, x::Real, var::Real)
    width = upper - lower
    # rescale
    μ = (x - lower) / width # ∈ [0, 1]
    σ2 = var * 1. / width^2

    # @assert σ^2 < μ * (1-μ) (var < width/2 necessary)

    α = ((1.0 - μ) / σ2 - 1/μ) * μ^2
    β = α * (1/μ - 1.)

    return width * Beta(α, β) + lower
end

function continuous_greater_than(lower::Real, x::Real, var::Real)
    μ = x - lower # ∈ [0, ∞)
    α = μ^2 / var
    θ = μ / α
    return Gamma(α, θ) + lower
end

# supports b = Inf
struct DiscreteRWProposer <: DiscreteUnivariateDistribution
    a::Int
    la::Float64
    b::Real
    lb::Float64
    μ::Int
    G::Distribution
    function DiscreteRWProposer(lower::Int, upper::Real, x::Int, var::Real)
        a = x - lower
        b = upper - x
        p = (sqrt(1 + 4*var) - 1) / (2*var)
        la = log(2*(1-(1-p)^a))
        lb = log(2*(1-(1-p)^b))
        return new(a, la, b, lb, x, Geometric(p))
    end
end

function Base.rand(d::DiscreteRWProposer)::Int
    if rand() > 0.5
        return rand(d.G) % d.b + d.μ + 1
    else
        return d.μ - rand(d.G) % d.a - 1
    end
end

function logpdf(d::DiscreteRWProposer, x::Int)::Float64
    y = x - d.μ
    if 0 < y <= d.b
        return logpdf(d.G, y-1) - d.lb
    elseif -d.a <= y < 0
        return logpdf(d.G, -y-1) - d.la
    end
    return -Inf
end

function discrete_interval(lower::Int, upper::Int, x::Int, var::Real)
    return DiscreteRWProposer(lower, upper, x, var)
end

function discrete_greater_than(lower::Real, x::Real, var::Real)
    return DiscreteRWProposer(lower, Inf, x, var)
end

export continuous_interval, continuous_greater_than, discrete_interval, discrete_greater_than

# using Plots
# d = continuous_interval(-1, 3, 1.5, 0.5)
# Distributions.mean(d)
# Distributions.var(d)
# plot(x -> exp(Distributions.logpdf(d, x)), xlims=(-1,3))


# d = continuous_greater_than(-1, 1.5, 0.5)
# Distributions.mean(d)
# Distributions.var(d)
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