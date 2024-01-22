import TinyPPL.Distributions: init_variational_distribution, logpdf_param_grads, get_params, update_params, to_unconstrained, MeanField
import Distributions

function get_elbo_and_grad_estimate!(pgm::PGM, X::Vector{Float64}, var_dists::Vector{<:VariationalDistribution}, l::Int, elbo::AbstractVector{Float64}, var_param_grads::Vector{Matrix{Float64}})
    for node in pgm.topological_order
        d = get_distribution(pgm, node, X)
        if isobserved(pgm, node)
            value = get_observed_value(pgm, node)
            elbo[node] = logpdf(d, value)
        else
            var_dist = var_dists[node]
            if d isa Distributions.ContinuousDistribution
                unconstrained_value = rand(var_dist)
                var_param_grads[node][:,l] = logpdf_param_grads(var_dist, unconstrained_value)
                transformed_dist = to_unconstrained(d)
                elbo[node] = logpdf(transformed_dist, unconstrained_value) - logpdf(var_dist, unconstrained_value)
                constrained_value = transformed_dist.T_inv(unconstrained_value)
                value = constrained_value
            else
                value = rand(var_dist)
                var_param_grads[node][:,l] = logpdf_param_grads(var_dist, value)
                elbo[node] = logpdf(d, value) - logpdf(var_dist, value)
            end
            X[node] = value
        end
    end
end

function bbvi_naive(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    K = pgm.n_latents
    X = Vector{Float64}(undef, K)
    pgm.sample!(X)

    var_dists = [init_variational_distribution(get_distribution(pgm, node, X)) for node in 1:K]
    
    eps = 1e-8
    acc = [fill(eps, length(get_params(var_dists[node]))) for node in 1:K]
    pre = 1.1
    post = 0.9

    elbos = zeros(L)
    _elbo = zeros(pgm.n_variables)

    @progress for _ in 1:n_samples
        var_param_grads = [zeros(length(get_params(var_dists[node])), L) for node in 1:K]
        for l in 1:L
            elbo = 0.
            get_elbo_and_grad_estimate!(pgm, X, var_dists, l, _elbo, var_param_grads)
            elbos[l] = sum(_elbo)
        end

        # Normal ReinforceELBO
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

    return GraphVIResult(pgm, MeanField(var_dists))
end

export bbvi_naive


function bbvi_rao(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    K = pgm.n_latents
    X = Vector{Float64}(undef, K)
    pgm.sample!(X)

    var_dists = [init_variational_distribution(get_distribution(pgm, node, X)) for node in 1:K]
    
    eps = 1e-8
    acc = [fill(eps, length(get_params(var_dists[node]))) for node in 1:K]
    pre = 1.1
    post = 0.9

    nodes = [n => [child for (x,child) in pgm.edges if x == n] for n in pgm.topological_order if !isobserved(pgm, n)]

    elbos = zeros(pgm.n_variables, L)

    @progress for i in 1:n_samples
        var_param_grads = [zeros(length(get_params(var_dists[node])), L) for node in 1:K]
        for l in 1:L
            _elbo = view(elbos, :, l)
            get_elbo_and_grad_estimate!(pgm, X, var_dists, l, _elbo, var_param_grads)
        end

        # rao-blackwellized gradient estimate
        for (node, children) in nodes
            # we only have to sum elbo from node and its children as opposed to all nodes
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
    
    return GraphVIResult(pgm, MeanField(var_dists))
end

export bbvi_rao