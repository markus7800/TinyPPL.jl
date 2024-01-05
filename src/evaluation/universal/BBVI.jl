import Distributions
import TinyPPL.Distributions: init_variational_distribution, logpdf_param_grads

mutable struct BBVI <: UniversalSampler
    ELBO::Float64 # log P(X,Y) - log Q(X; λ)
    variational_dists::Dict{Any, VariationalDistribution}
    variational_param_grads::Dict{Any, AbstractVector{Float64}}

    function BBVI(variational_dists)
        return new(0., variational_dists, Dict{Any, AbstractVector{Float64}}())
    end
end

function sample(sampler::BBVI, addr::Any, dist::Distributions.DiscreteDistribution, obs::Union{Nothing, Real})::Real
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

function sample(sampler::BBVI, addr::Any, dist::Distributions.ContinuousDistribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.ELBO += logpdf(dist, obs)
        return obs
    end

    if !haskey(sampler.variational_dists, addr)
        # assumes static distribution type, support of continuous distributions may be dynamic
        sampler.variational_dists[addr] = init_variational_distribution(dist)
    end
    var_dist = sampler.variational_dists[addr]
    unconstrained_value = rand(var_dist)
    lpq = logpdf(var_dist, unconstrained_value)
    sampler.variational_param_grads[addr] = logpdf_param_grads(var_dist, unconstrained_value)
    transformed_dist = to_unconstrained(dist)
    sampler.ELBO += logpdf(transformed_dist, unconstrained_value) - lpq
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

function bbvi(model::UniversalModel, args::Tuple, observations::Dict,  n_samples::Int, L::Int, learning_rate::Float64, variational_dists = Dict{Any, VariationalDistribution}())

    eps = 1e-8
    acc = Dict{Any, AbstractVector{Float64}}()
    pre = 1.1
    post = 0.9
    
    @progress for i in 1:n_samples

        samplers = [BBVI(variational_dists) for _ in 1:L]
        for i in 1:L
            sampler = samplers[i]
            _ = model(args, sampler, observations)
        end


        for (addr, var_dist) in variational_dists
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

            acc_addr = get(acc, addr, fill(eps,size(grad)))
            acc_addr = post .* acc_addr .+ pre .* grad.^2
            acc[addr] = acc_addr
            rho = learning_rate ./ (sqrt.(acc_addr) .+ eps)

            # update params
            # params = no_grad(params)
            params += rho .* grad
            variational_dists[addr] = update_params(var_dist, params)
        end
        
    end

    function _transform_to_constrained(X::Dict{Any,Float64})
        sampler = UniversalConstraintTransformer(X, :constrained)
        model(args, sampler, observations)
        return sampler.Y
    end
    return UniversalVIResult(UniversalMeanField(variational_dists), _transform_to_constrained)
end

export bbvi