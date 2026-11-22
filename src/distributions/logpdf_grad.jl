import SpecialFunctions: digamma

# TODO: rewrite to VariationalWrappedDistribution
# NOTE: currently unused (BBVI uses AD), kept for reference.

# parameter gradients for BBVI

function logpdf_params_grad(d::Distribution, x::Real)
    error("logpdf_grad not implemented for $d.")
end

function constrain(d::Distribution, x::Tuple)
    error("constrain not implemented for $d $x.")
end

# if gradient does not exist for parameter p , then set it to 0.
# such that p + λ∇ = p

# discrete methods take x::Real, since values are not always Int (e.g. Bernoulli samples are Bool, traces store Float64)
function logpdf_params_grad(d::Bernoulli, x::Real)
    ∇p = x == 1 ? 1. / d.p : -1. / (1-d.p)
    return (∇p,)
end
function constrain(::Bernoulli, params::Tuple{Float64})
    p = params[1]
    return (clamp(p, 0., 1.), )
end

function logpdf_params_grad(d::Binomial, x::Real)
    ∇n = 0. # n is discrete
    ∇p = (x / d.p - (d.n - x) / (1 - d.p))
    return (∇n, ∇p)
end
function constrain(::Binomial, params::Tuple{Real,Real})
    n, p = params
    return (n, clamp(p, 0., 1.))
end

function logpdf_params_grad(d::Categorical, x::Real)
    x = Int(x)
    ∇p = zeros(length(d.p))
    ∇p[x] = 1. / d.p[x]
    return (∇p,)
end

# function logpdf_params_grad(d::DiscreteUniform, x::Int)
#     return (0,0)
# end

function logpdf_params_grad(d::Geometric, x::Real)
    ∇p = x >= 0 ? 1/d.p - (1/(1-d.p) * x) : 0.
    return (∇p,)
end

function logpdf_params_grad(d::Poisson, x::Real)
    ∇p = x/d.λ - 1
    return (∇p,)
end

function logpdf_params_grad(d::Beta, x::Real)
    @assert 0. ≤ x ≤ 1.
    ∇α = log(x) - (digamma(d.α) - digamma(d.α + d.β))
    ∇β = log1p(-x) - (digamma(d.β) - digamma(d.α + d.β))
    return (∇α, ∇β)
end

function logpdf_params_grad(d::Cauchy, x::Real)
    x_μ = x - d.μ
    x_μ_sq = x_μ^2
    σ_sq = d.σ^2
    ∇μ =  2 * x_μ / (σ_sq + x_μ_sq)
    ∇σ = (x_μ_sq - σ_sq) / (d.σ * (σ_sq + x_μ_sq))
    return (∇μ, ∇σ)
end

# Exponential is parametrised by scale θ: logpdf = -log(θ) - x/θ
function logpdf_params_grad(d::Exponential, x::Real)
    ∇θ = -1/d.θ + x/d.θ^2
    return (∇θ,)
end

# shape = α
# scale = θ
function logpdf_params_grad(d::Gamma, x::Real)
    if x > 0.
        ∇α = log(x) - log(d.θ) - digamma(d.α)
        ∇θ = x / (d.θ * d.θ) - (d.α / d.θ)
        return (∇α, ∇θ)
    else
        return (0., 0.)
    end
end

function logpdf_params_grad(d::InverseGamma, x::Real)
    α, θ = params(d) # InverseGamma stores Gamma(α, 1/θ) internally, no fields α, θ
    ∇α = log(θ) - digamma(α) - log(x)
    ∇θ = α / θ - 1/x
    return (∇α, ∇θ)
end

function logpdf_params_grad(d::Laplace, x::Real)
    precision = 1. / d.θ
    if x > d.μ
        diff = -1.
    else
        diff = 1.
    end
    ∇μ = -(diff * precision)
    ∇θ = -1/d.θ + abs(x-d.μ) / (d.θ * d.θ)
    return (∇μ, ∇θ)
end

function logpdf_params_grad(d::LogNormal, x::Real)
    z = (log(x) - d.μ) / d.σ
    ∇μ = z / d.σ
    ∇σ = (z - 1.) * (z + 1.) / d.σ
    return (∇μ, ∇σ)
end

function logpdf_params_grad(d::Normal, x::Real)
    z = (x - d.μ) / d.σ
    ∇μ = z / d.σ
    ∇σ = -1. / d.σ + abs2(z) / d.σ
    return (∇μ, ∇σ)
end

function logpdf_params_grad(d::TDist, x::Real)
    x_sq = x^2
    a = x_sq / d.ν + 1

    l = (x_sq + x_sq/d.ν) / (2*d.ν * a)
    r = -0.5 * (log(a) + 1/d.ν + digamma(d.ν/2) - digamma((d.ν+1)/2))

    ∇ν = l + r
    return (∇ν, )
end

function logpdf_params_grad(d::Uniform, x::Real)
    if d.a ≤ x ≤ d.b
        ∇ = 1. / (d.b - d.a)
        return (∇, -∇)
    else
        return (0., 0.)
    end
end

export logpdf_params_grad
