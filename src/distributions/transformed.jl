import Distributions: Distribution, ContinuousUnivariateDistribution, logpdf, support, RealInterval
export RealInterval

"""
This file includes several f:ℝ → ℝ bijective transformations.
The transformations implement a method to compute the log_abs_det_jacobian = log |d/dx f(x)|.
There is also utilities to compute the support and image of transformations.
"""

abstract type Transform end

function (::Transform)(x::Real)::Real
    error("Not implemented.")
end
function log_abs_det_jacobian(t::Transform, x::Real)::Real
    error("Not implemented.")
end
function Base.inv(t::Transform)::Transform
    error("Not implemented.")
end
# x is in the domain of t
function in_domain(t::Transform, x::Real)::Bool
    error("Not implemented.")
end
# domain of t as interval
function domain(t::Transform)::RealInterval
    error("Not implemented.")
end
# image of interval `domain` (⊆ domain(t)) under t, all transforms are monotonic
function image(t::Transform, domain::RealInterval)::RealInterval
    error("Not implemented.")
end

struct IdentityTransform <: Transform
end
function (::IdentityTransform)(x::Real)::Real
    return x
end
function log_abs_det_jacobian(t::IdentityTransform, x::Real)::Real
    return 0.
end
function Base.inv(t::IdentityTransform,)::Transform
    return t
end
in_domain(t::IdentityTransform, x::Real) = true
domain(t::IdentityTransform)::RealInterval = RealInterval(-Inf, Inf)
image(t::IdentityTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))


struct ExpTransform <: Transform
end
export ExpTransform
struct LogTransform <: Transform
end
export LogTransform
function (::ExpTransform)(x::Real)::Real
    return exp(x)
end
function log_abs_det_jacobian(t::ExpTransform, x::Real)::Real
    return x
end
function Base.inv(t::ExpTransform,)::Transform
    return LogTransform()
end
in_domain(t::ExpTransform, x::Real) = true
domain(t::ExpTransform)::RealInterval = RealInterval(-Inf, Inf)
image(t::ExpTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))

function (::LogTransform)(x::Real)::Real
    return log(x)
end
function log_abs_det_jacobian(t::LogTransform, x::Real)::Real
    return -log(abs(x))
end
function Base.inv(t::LogTransform,)::Transform
    return ExpTransform()
end
in_domain(t::LogTransform, x::Real) = 0 < x
domain(t::LogTransform) = RealInterval(0, Inf)
image(t::LogTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))

struct SigmoidTransform <: Transform
end
export SigmoidTransform
struct InverseSigmoidTransform <: Transform
end
export InverseSigmoidTransform
function (::SigmoidTransform)(x::Real)::Real
    return 1 / (1 + exp(-x))
end
function log_abs_det_jacobian(t::SigmoidTransform, x::Real)::Real
    # log σ'(x) = log σ(x) + log σ(-x) = -log1p(exp(-x)) - log1p(exp(x)), symmetric in x.
    # x - 2 * log1p(exp(x)) overflows for large x, written with -|x| it does not.
    return -abs(x) - 2 * log1p(exp(-abs(x)))
end
function Base.inv(t::SigmoidTransform,)::Transform
    return InverseSigmoidTransform()
end
in_domain(t::SigmoidTransform, x::Real) = true
domain(t::SigmoidTransform) = RealInterval(-Inf, Inf)
image(t::SigmoidTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))

function (::InverseSigmoidTransform)(x::Real)::Real
    return log(x / (1-x))
end
function log_abs_det_jacobian(t::InverseSigmoidTransform, x::Real)::Real
    # -log(x(1-x)), with log1p for precision near x=1 (x-x^2 cancels)
    return -log(x) - log1p(-x)
end
function Base.inv(t::InverseSigmoidTransform,)::Transform
    return SigmoidTransform()
end
in_domain(t::InverseSigmoidTransform, x::Real) = 0 < x && x < 1
domain(t::InverseSigmoidTransform) = RealInterval(0, 1)
image(t::InverseSigmoidTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))

struct AffineTransform{T<:Real} <: Transform
    k::T
    d::T
end
AffineTransform(k::Real, d::Real) = AffineTransform(promote(k, d)...)
export AffineTransform
function (t::AffineTransform)(x::Real)::Real
    return t.k * x + t.d
end
function log_abs_det_jacobian(t::AffineTransform, x::Real)::Real
    return log(abs(t.k))
end
function Base.inv(t::AffineTransform,)::Transform
    return AffineTransform(1/t.k, -t.d/t.k)
end
in_domain(t::AffineTransform, x::Real) = true
domain(t::AffineTransform) = RealInterval(-Inf, Inf)
image(t::AffineTransform, domain::RealInterval) = t.k > 0 ? RealInterval(t(domain.lb), t(domain.ub)) :  RealInterval(t(domain.ub), t(domain.lb))

# composition of two transforms where t(x) = t2(t1(x))
struct ComposeTransform{T1<:Transform,T2<:Transform} <: Transform
    t1::T1
    t2::T2
end
export ComposeTransform
function (t::ComposeTransform)(x::Real)::Real
    return t.t2(t.t1(x))
end
function Base.:∘(t2::Transform, t1::Transform)::ComposeTransform
    return ComposeTransform(t1, t2)
end
function log_abs_det_jacobian(t::ComposeTransform, x::Real)::Real
    x1 = x
    x2 = t.t1(x)
    return log_abs_det_jacobian(t.t2, x2) + log_abs_det_jacobian(t.t1, x1)
end
function Base.inv(t::ComposeTransform,)::Transform
    return inv(t.t1) ∘ inv(t.t2)
end
in_domain(t::ComposeTransform, x::Real) = in_domain(t.t1, x) && in_domain(t.t2, t.t1(x))
intersect_intervals(a::RealInterval, b::RealInterval) = RealInterval(max(a.lb, b.lb), min(a.ub, b.ub))
# x ∈ domain(t1) with t1(x) ∈ domain(t2)
function domain(t::ComposeTransform)
    range_t1 = image(t.t1, domain(t.t1))
    # preimage of domain(t2) under t1 (t1 is a bijection from domain(t1) to range_t1)
    return image(inv(t.t1), intersect_intervals(domain(t.t2), range_t1))
end
# t1 or t2 may be decreasing (e.g. AffineTransform with k < 0)
image(t::ComposeTransform, domain::RealInterval) = image(t.t2, image(t.t1, domain))

"""
if X ~ base then for Y=TransformedDistribution(base,T) Y ~ T(X)
This wrapper essentially implements the probability density transform.
"""
struct TransformedDistribution{D<:ContinuousUnivariateDistribution,F<:Transform,G<:Transform} <: ContinuousUnivariateDistribution
    base::D
    T::F
    T_inv::G
    function TransformedDistribution(base::D, T::F) where {D<:ContinuousUnivariateDistribution,F<:Transform}
        T_inv = inv(T)
        return new{D,F,typeof(T_inv)}(base, T, T_inv)
    end
end
export TransformedDistribution

function Base.rand(t::TransformedDistribution)::Real
    return t.T(Base.rand(t.base))
end

function Base.rand(t::TransformedDistribution, n::Int)::Vector{<:Real}
    return t.T.(Base.rand(t.base, n))
end

function logpdf(t::TransformedDistribution, y::Real)::Real
    if !in_domain(t.T_inv, y) # !(y in support(t))
        return -Inf
    end
    # probability density transform
    x = t.T_inv(y)
    return logpdf(t.base, x) + log_abs_det_jacobian(t.T_inv, y)
end

function support(t::TransformedDistribution)
    return image(t.T, support(t.base))
end

"""
Returns a transfomr that maps the given interval `supp` bijectively to ℝ
"""
function transform_to_unconstrained_from(supp::RealInterval)::Transform
    if supp.lb == -Inf && supp.ub == Inf
        return IdentityTransform()
    end

    if supp.ub == Inf # supp.lb != -Inf
        if supp.lb == 0
            return LogTransform()
        else
            return LogTransform() ∘ AffineTransform(1, -supp.lb)
        end
    end

    if supp.lb == -Inf # supp.ub != Inf
        return LogTransform() ∘ AffineTransform(-1, supp.ub) # x ↦ log(ub - x)
    end

    if supp.lb == 0 && supp.ub == 1
        return InverseSigmoidTransform()
    else
        scale = supp.ub - supp.lb
        return InverseSigmoidTransform() ∘ AffineTransform(1/scale, -supp.lb/scale)
    end

end

"""
Returns the transoform that maps ℝ bijectively to `interval`.
"""
function transform_to(interval::RealInterval)::Transform
    return inv(transform_to_unconstrained_from(interval))
end

"""
Transforms a (constrained) distribution to an unconstrained distribution.
"""
function to_unconstrained(base::ContinuousUnivariateDistribution)::ContinuousUnivariateDistribution
    supp = support(base)
    transform = transform_to_unconstrained_from(supp)
    return TransformedDistribution(base, transform)
end

export transform_to_unconstrained_from, transform_to, to_unconstrained, support
