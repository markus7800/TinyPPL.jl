import Distributions: Distribution, ContinuousUnivariateDistribution, logpdf, support, RealInterval

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
function in_support(t::Transform, x::Real)::Bool
    error("Not implemented.")
end
function support(t::Transform)::RealInterval
    error("Not implemented.")
end
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
struct LogTransform <: Transform
end
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
struct InverseSigmoidTransform <: Transform
end
function (::SigmoidTransform)(x::Real)::Real
    return 1 / (1 + exp(-x))
end
function log_abs_det_jacobian(t::SigmoidTransform, x::Real)::Real
    return x - 2 * log1p(exp(x))
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
    return -log(abs(x-x^2))
end
function Base.inv(t::InverseSigmoidTransform,)::Transform
    return SigmoidTransform()
end
in_domain(t::InverseSigmoidTransform, x::Real) = 0 < x && x < 1
domain(t::InverseSigmoidTransform) = RealInterval(0, 1)
image(t::InverseSigmoidTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))

struct AffineTransform <: Transform
    k::Real
    d::Real
end
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


struct ComposeTransform <: Transform
    t1::Transform
    t2::Transform
end
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
domain(t::ComposeTransform) = domain(t.t1)
image(t::ComposeTransform, domain::RealInterval) = RealInterval(t(domain.lb), t(domain.ub))


struct TransformedDistribution <: ContinuousUnivariateDistribution
    base::ContinuousUnivariateDistribution
    T::Transform
    T_inv::Transform
    function TransformedDistribution(base::ContinuousUnivariateDistribution, T::Transform)
        return new(base, T, inv(T))
    end
end

function Base.rand(t::TransformedDistribution)::Real
    return t.T(Base.rand(t.base))
end

function logpdf(t::TransformedDistribution, y::Real)::Real
    if !in_domain(t.T_inv, y) # !(y in support(t))
        return -Inf
    end
    x = t.T_inv(y)
    return logpdf(t.base, x) + log_abs_det_jacobian(t.T_inv, y)
end

function support(t::TransformedDistribution)
    return image(t.T, support(t.base))
end


function transform_to(supp::RealInterval)::Transform
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
        return LogTransform() ∘ AffineTransform(-1, supp.lb)
    end

    if supp.lb == 0 && supp.ub == 1
        return InverseSigmoidTransform()
    else
        scale = supp.ub - supp.lb
        return InverseSigmoidTransform() ∘ AffineTransform(1/scale, -supp.lb/scale)
    end

end
function to_unconstrained(base::ContinuousUnivariateDistribution)::ContinuousUnivariateDistribution
    supp = support(base)
    transform = transform_to(supp)
    return TransformedDistribution(base, transform)
end

export transform_to, to_unconstrained


# using Distributions
# using Plots

# to_unconstrained(Normal(0,1))

# d_constrained = Gamma(1,1)
# d_unconstrained = to_unconstrained(d_constrained)
# d_unconstrained = TransformedDistribution(d_constrained, LogTransform())

# d_constrained = Beta(2,3)
# d_unconstrained = to_unconstrained(d_constrained)
# d_unconstrained = TransformedDistribution(d_constrained, InverseSigmoidTransform())


# d_constrained = Uniform(2,4)
# d_unconstrained = to_unconstrained(d_constrained)
# d_unconstrained = TransformedDistribution(d_constrained, InverseSigmoidTransform() ∘ AffineTransform(1/(4-2),-2/2))

# xs = [rand(d_unconstrained) for _ in 1:100000]
# histogram(xs, normalize=true)
# plot!(x -> exp(logpdf(d_unconstrained, x)), linewidth=3)

# d = TransformedDistribution(d_unconstrained, d_unconstrained.T_inv)
# plot(x -> exp(logpdf(d_constrained, x)))
# plot!(x -> exp(logpdf(d, x)))

