import Tracker

function advi(n_samples::Int, K::Int, L::Int, learning_rate::Float64, logjoint::Function)
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

export advi