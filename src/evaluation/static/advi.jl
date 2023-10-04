import Tracker
import Distributions
import LinearAlgebra

function advi(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, K::Int)
    mu = zeros(K)
    omega = zeros(K)
    phi = vcat(mu, omega)

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi = Tracker.param(phi)
        mu = phi[1:K]
        omega = phi[K+1:end]

        # reparametrisation trick
        eta = randn(K)
        zeta = @. exp(omega) * eta + mu

        # estimate elbo
        elbo = 0.
        for _ in 1:L
            elbo += logjoint(zeta)
        end
        elbo = elbo / L + sum(omega) # + K/2 * (log(2π) + 1)

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi)

        # reset from gradient computation
        phi = Tracker.data(phi)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    mu = phi[1:K]
    omega = phi[K+1:end]
    return mu, exp.(omega)
end


abstract type VariationalDistribution <: Distribution{Distributions.Multivariate, Distributions.Continuous} end

function update_q(q::VariationalDistribution, phi::AbstractVector{<:Float64})::VariationalDistribution
    error("Not implemented.")
end
function nparams(q::VariationalDistribution)
    return sum(length, Distributions.params(q.base))
end
function Base.rand(q::VariationalDistribution)
    return rand(q.base)
end
function Base.rand(q::VariationalDistribution, n::Int)
    return rand(q.base, n)
end
function Distributions.entropy(q::VariationalDistribution)
    return Distributions.entropy(q.base)
end   

# struct MeanFieldGaussian <: VariationalDistribution
#     base::Distributions.MultivariateNormal
# end

# function MeanFieldGaussian(K::Int)
#     return MeanFieldGaussian(Distributions.MultivariateNormal(zeros(K), LinearAlgebra.diagm(ones(K))))
# end

# function update_q(q::MeanFieldGaussian, phi::AbstractVector{<:Float64})::VariationalDistribution
#     K = length(q.base)
#     mu = phi[1:K]
#     omega = phi[K+1:end]
#     return MeanFieldGaussian(Distributions.MultivariateNormal(mu, LinearAlgebra.diagm(exp.(omega))))
# end
# function nparams(q::MeanFieldGaussian)
#     return 2*length(q.base)
# end

struct MeanFieldGaussian <: VariationalDistribution
    mu::AbstractVector{<:Real}
    sigma::AbstractVector{<:Real}
end

function MeanFieldGaussian(K::Int)
    return MeanFieldGaussian(zeros(K), ones(K))
end

function update_q(q::MeanFieldGaussian, phi::AbstractVector{<:Float64})::VariationalDistribution
    K = length(q.mu)
    mu = phi[1:K]
    omega = phi[K+1:end]
    return MeanFieldGaussian(mu, exp.(omega))
end
function nparams(q::MeanFieldGaussian)
    return 2*length(q.mu)
end
function Base.rand(q::MeanFieldGaussian)
    return q.sigma .* randn(length(q.mu)) .+ q.mu
end
# function Base.rand(q::VariationalDistribution, n::Int)
#     return rand(q.base, n)
# end

function Distributions.entropy(q::VariationalDistribution)
    return sum(log, q.sigma) + length(q.mu)/2 * (log(2π) + 1)
end   

function advi(logjoint::Function, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution)
    phi = zeros(nparams(q))

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    @progress for i in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        q = update_q(q, phi_tracked)

        # implicit reparametrisation trick (if we get gradients)
        zeta = rand(q)

        # estimate elbo
        elbo = 0.
        for _ in 1:L
            elbo += logjoint(zeta)
        end
        elbo = elbo / L + Distributions.entropy(q)

        # automatically compute gradient
        Tracker.back!(elbo)
        grad = Tracker.grad(phi_tracked)

        # decayed adagrad update rule
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    return update_q(q, phi)
end

export advi, MeanFieldGaussian