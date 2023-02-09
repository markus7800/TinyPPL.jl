module Distributions
    import Distributions: Distribution, logpdf, params

    import Distributions: Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
    import Distributions: Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform

    import SpecialFunctions: digamma

    # parameter gradients for BBVI

    function logpdf_params_grad(d::Distribution, x::Real)
        error("logpdf_grad not implemented for $d.")
    end

    function constrain(::Distribution, x::Tuple)
        error("constrain not implemented for $d $x.")
    end

    # if gradient does not exist for parameter p , then set it to 0.
    # such that p + λ∇ = p

    function logpdf_params_grad(d::Bernoulli, x::Int)
        ∇p = x == 1 ? 1. / d.p : -1. / (1-d.p)
        return (∇p,)
    end
    function constrain(::Bernoulli, params::Tuple{Float64})
        p = params[1]
        return (clamp(p, 0., 1.), )
    end

    function logpdf_params_grad(d::Binomial, x::Int)
        ∇n = 0
        ∇p = (x / d.p - (d.n - x) / (1 - d.p))
        return (∇n, ∇p)
    end
    function constrain(::Binomial, params::Tuple{Float64})
        n, p = params
        return (n, clamp(p, 0., 1.))
    end

    function logpdf_params_grad(d::Categorical, x::Int)
        ∇p = zeros(length(d.p))
        ∇p[x] = 1. / d.p[x]
        return (∇p,)
    end

    function logpdf_params_grad(d::DiscreteUniform, x::Int)
        return (0,0)
    end

    function logpdf_params_grad(d::Geometric, x::Int)
        ∇p = x >= 0 ? 1/d.p - (1/(1-d.p) * x) : 0.
        return (∇p,)
    end

    function logpdf_params_grad(d::Poisson, x::Int)
        ∇p = x/d.λ - 1
        return (∇p,)
    end

    function logpdf_params_grad(d::Beta, x::Float64)
        @assert 0. ≤ x ≤ 1.
        ∇α = log(x) - (digamma(d.α) - digamma(d.α + d.β))
        ∇β = log1p(-x) - (digamma(d.β) - digamma(d.α + d.β))
        return (∇α, ∇β)
    end

    function logpdf_params_grad(d::Cauchy, x::Float64)
        x_μ = x - d.μ
        x_μ_sq = x_μ^2
        σ_sq = d.σ^2
        ∇μ =  2 * x_μ / (σ_sq + x_μ_sq)
        ∇σ = (x_μ_sq - σ_sq) / (d.σ * (σ_sq + x_μ_sq))
        return (∇μ, ∇σ)
    end

    function logpdf_params_grad(d::Exponential, x::Float64)
        ∇θ = 1/d.θ - x
        return (∇θ,)
    end

    # shape = α
    # scale = θ
    function logpdf_params_grad(d::Gamma, x::Float64)
        if x > 0.
            ∇α = log(x) - log(d.θ) - digamma(d.α)
            ∇θ = x / (d.θ * d.θ) - (d.α / d.θ)
            return (∇α, ∇θ)
        else
            return (0., 0.)
        end
    end

    function logpdf_params_grad(d::InverseGamma, x::Float64)
        ∇α = log(d.θ) - SpecialFunctions.digamma(d.α) - log(x)
        ∇θ = d.α / d.θ - 1/x
        return (∇α, ∇θ)
    end

    function logpdf_params_grad(d::Laplace, x::Float64)
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

    function logpdf_params_grad(d::LogNormal, x::Float64)
        z = (log(x) - d.μ) / d.σ
        ∇μ = z / d.σ
        ∇σ = (z - 1.) * (z + 1.) / d.σ
        return (∇μ, ∇σ)
    end

    function logpdf_params_grad(d::Normal, x::Float64)
        z = (x - d.μ) / d.σ
        ∇μ = z / d.σ
        ∇σ = -1. / d.σ + abs2(z) / d.σ
        return (∇μ, ∇σ)
    end

    function logpdf_params_grad(d::TDist, x::Float64)
        x_sq = x^2
        a = x_sq / d.ν + 1

        l = (x_sq + x_sq/d.ν) / (2*d.ν * a)
        r = -0.5 * (log(a) + 1/d.ν + digamma(d.ν/2) - digamma((d.ν+1)/2))

        ∇ν = l + r
        return (∇ν, )
    end

    function logpdf_params_grad(d::Uniform, x::Float64)
        if d.a ≤ x ≤ d.b
            ∇ = 1. / (d.b - d.a)
            return (∇, -∇)
        else
            return (0., 0.)
        end
    end


    export Distribution, logpdf, params

    export Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
    export Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform

    export logpdf_params_grad
end