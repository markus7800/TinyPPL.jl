import ..Distributions: init_variational_distribution, logpdf_param_grads, get_params, update_params, to_unconstrained, MeanField
import Distributions

function bbvi_naive(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample!(X)

    observed = .!isnothing.(pgm.observed_values)
    K = pgm.n_variables - sum(observed)
    @assert !any(observed[1:K])
    @assert all(observed[K+1:end])

    var_dists = [init_variational_distribution(pgm.distributions[node](X)) for node in 1:K]
    
    eps = 1e-8
    acc = [fill(eps, length(get_params(var_dists[node]))) for node in 1:K]
    pre = 1.1
    post = 0.9

    @progress for _ in 1:n_samples
        var_param_grads = [zeros(length(get_params(var_dists[node])), L) for node in 1:K]
        elbos = zeros(L)
        for l in 1:L
            elbo = 0.
            for node in pgm.topological_order
                d = pgm.distributions[node](X)
                if observed[node]
                    value = pgm.observed_values[node](X)
                    elbo += logpdf(d, value)
                else
                    var_dist = var_dists[node]
                    if d isa Distributions.ContinuousDistribution
                        unconstrained_value = rand(var_dist)
                        var_param_grads[node][:,l] = logpdf_param_grads(var_dist, unconstrained_value)
                        transformed_dist = to_unconstrained(d)
                        elbo += logpdf(transformed_dist, unconstrained_value) - logpdf(var_dist, unconstrained_value)
                        constrained_value = transformed_dist.T_inv(unconstrained_value)
                        value = constrained_value
                    else
                        value = rand(var_dist)
                        var_param_grads[node][:,l] = logpdf_param_grads(var_dist, value)
                        elbo += logpdf(d, value) - logpdf(var_dist, value)
                    end
                end
                X[node] = value
            end
            elbos[l] = elbo
        end

        for node in 1:K
            grad = vec(sum(reshape(elbos,1,:) .* var_param_grads[node],dims=2)) ./ L
            acc_node = acc[node]
            acc_node = post .* acc_node .+ pre .* grad.^2
            acc[node] = acc_node
            rho = learning_rate ./ (sqrt.(acc_node) .+ eps)

            var_dist = var_dists[node]
            params = get_params(var_dist)
            params += rho .* grad
            var_dists[node] = update_params(var_dist, params)
        end
    end

    return MeanField(var_dists)
end
export bbvi_naive


function bbvi_rao(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample!(X)

    observed = .!isnothing.(pgm.observed_values)
    K = pgm.n_variables - sum(observed)
    @assert !any(observed[1:K])
    @assert all(observed[K+1:end])


    var_dists = [init_variational_distribution(pgm.distributions[node](X)) for node in 1:K]
    
    eps = 1e-8
    acc = [fill(eps, length(get_params(var_dists[node]))) for node in 1:K]
    pre = 1.1
    post = 0.9

    nodes = [n => [child for (x,child) in pgm.edges if x == n] for n in pgm.topological_order if !observed[n]]

    @progress for i in 1:n_samples
        var_param_grads = [zeros(length(get_params(var_dists[node])), L) for node in 1:K]
        elbos = zeros(pgm.n_variables, L)
        for l in 1:L
            for node in pgm.topological_order
                d = pgm.distributions[node](X)
                if observed[node]
                    value = pgm.observed_values[node](X)
                    elbos[node,l] = logpdf(d, value)
                else
                    var_dist = var_dists[node]
                    if d isa Distributions.ContinuousDistribution
                        unconstrained_value = rand(var_dist)
                        var_param_grads[node][:,l] = logpdf_param_grads(var_dist, unconstrained_value)
                        transformed_dist = to_unconstrained(d)
                        elbos[node,l] = logpdf(transformed_dist, unconstrained_value) - logpdf(var_dist, unconstrained_value)
                        constrained_value = transformed_dist.T_inv(unconstrained_value)
                        value = constrained_value
                    else
                        value = rand(var_dist)
                        var_param_grads[node][:,l] = logpdf_param_grads(var_dist, value)
                        elbos[node,l] = logpdf(d, value) - logpdf(var_dist, value)
                    end
                end
                X[node] = value
            end
        end

        # rao-blackwellized gradient estimate
        for (node, children) in nodes
            rao_elbos = elbos[node,:]
            for child in children
                rao_elbos += elbos[child,:]
            end
            grad = vec(sum(reshape(rao_elbos,1,:) .* var_param_grads[node],dims=2)) ./ L
            acc_node = acc[node]
            acc_node = post .* acc_node .+ pre .* grad.^2
            acc[node] = acc_node
            rho = learning_rate ./ (sqrt.(acc_node) .+ eps)

            var_dist = var_dists[node]
            params = get_params(var_dist)
            params += rho .* grad
            var_dists[node] = update_params(var_dist, params)
        end
    end
    
    return MeanField(var_dists)
end

export bbvi_rao