import Distributions
import TinyPPL.Distributions: init_variational_distribution, logpdf_param_grads

mutable struct BBVI <: UniversalSampler
    ELBO::Float64 # log P(X,Y) - log Q(X; ψ)
    variational_dists::Dict{Address, VariationalDistribution}
    variational_param_grads::Dict{Address, Vector{Float64}}

    function BBVI(variational_dists)
        return new(0., variational_dists, Dict{Any, AbstractVector{Float64}}())
    end
end

function sample(sampler::BBVI, addr::Address, dist::Distributions.DiscreteDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.ELBO += logpdf(dist, obs)
        return obs
    end

    if !haskey(sampler.variational_dists, addr)
        # assumes static distribution type, support of continuous distributions may be dynamic
        sampler.variational_dists[addr] = init_variational_distribution(dist)
    end
    var_dist = sampler.variational_dists[addr]
    value = rand(var_dist)
    lpq = logpdf(var_dist, value)
    sampler.variational_param_grads[addr] = logpdf_param_grads(var_dist, value)
    sampler.ELBO += logpdf(dist, value) - lpq
    return value
end

function sample(sampler::BBVI, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.ELBO += logpdf(dist, obs)
        return obs
    end

    if !haskey(sampler.variational_dists, addr)
        # assumes static distribution type, support of continuous distributions may be dynamic
        sampler.variational_dists[addr] = init_variational_distribution(dist)
    end
    # fit to unconstrained model
    var_dist = sampler.variational_dists[addr]
    unconstrained_value = rand(var_dist)
    lpq = logpdf(var_dist, unconstrained_value)
    sampler.variational_param_grads[addr] = logpdf_param_grads(var_dist, unconstrained_value)
    transformed_dist = to_unconstrained(dist)
    sampler.ELBO += logpdf(transformed_dist, unconstrained_value) - lpq
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

"""
BBVI, where we fit unconstrained model.
VariationalDistribution may be provided but are defaulted to init_variational_distribution(dist).
Thus, the distribution type is assumed to be constant.
ELBO gradient is approximated with the REINFORCE method.
No automatic differentiation is used, instead we use logpdf_param_grads method of VariationalDistribution.
Thus, also discrete latent variables are allowed.
"""
function bbvi(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64, variational_dists = Dict{Address,VariationalDistribution}())

    eps = 1e-8
    acc = Dict{Any, AbstractVector{Float64}}()
    pre = 1.1
    post = 0.9
    
    @progress for i in 1:n_samples

        # variational_dists is shared amont all samplers
        samplers = [BBVI(variational_dists) for _ in 1:L]
        for i in 1:L
            sampler = samplers[i]
            _ = model(args, sampler, observations)
        end


        for (addr, var_dist) in variational_dists
            # compute gradient
            params = get_params(var_dist)
            grad = zeros(size(params))
            for i in 1:L
                sampler = samplers[i]
                if !haskey(sampler.variational_param_grads, addr)
                    sampler.variational_param_grads[addr] = zeros(size(params))
                end
                # TODO: maybe control variate?
                grad .+= sampler.variational_param_grads[addr] .* sampler.ELBO
            end
            grad = grad / L

            # adagrad update
            acc_addr = get(acc, addr, fill(eps,size(grad)))
            acc_addr = post .* acc_addr .+ pre .* grad.^2
            acc[addr] = acc_addr
            rho = learning_rate ./ (sqrt.(acc_addr) .+ eps)

            # update params
            params += rho .* grad
            variational_dists[addr] = update_params(var_dist, params)
        end
        
    end

    _transform_to_constrained(X::AbstractUniversalTrace) = transform_to_constrained(X, model, args, observations)
    return UniversalVIResult(UniversalMeanField(variational_dists, model, args, observations), _transform_to_constrained)
end

export bbvi