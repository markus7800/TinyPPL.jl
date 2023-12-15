import Distributions
import ..Distributions: transform_to, support, IdentityTransform, RealInterval, logpdf_param_grads
import ..Distributions: VariationalNormal, VariationalGeometric

mutable struct BBVI <: UniversalSampler
    ELBO::Float64 # log P(X,Y) - log Q(X; λ)
    variational_dists::Dict{Any, VariationalDistribution}
    variational_param_grads::Dict{Any, AbstractVector{Float64}}

    function BBVI(variational_dists)
        return new(0., variational_dists, Dict{Any, AbstractVector{Float64}}())
    end
end

init_variational_distribution(::Distributions.ContinuousUnivariateDistribution) = VariationalNormal()
init_variational_distribution(::Distributions.Geometric) = VariationalGeometric()

function sample(sampler::BBVI, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.ELBO += logpdf(dist, obs)
        return obs
    end

    if !haskey(sampler.variational_dists, addr)
        # assumes static distribution type, support of continuous distributions may be dynamic
        sampler.variational_dists[addr] = init_variational_distribution(dist)
    end
    var_dist = sampler.variational_dists[addr]
    if dist isa Distributions.ContinuousUnivariateDistribution
        transformed_dist = to_unconstrained(dist)
        unconstrained_value = rand(var_dist)
        lpq = logpdf(var_dist, unconstrained_value)
        sampler.variational_param_grads[addr] = logpdf_param_grads(var_dist, unconstrained_value)
        sampler.ELBO += logpdf(transformed_dist, unconstrained_value) - lpq
        constrained_value = transformed_dist.T_inv(unconstrained_value)
        return constrained_value
    else
        @assert dist isa Distributions.DiscreteUnivariateDistribution dist
        value = rand(var_dist)
        lpq = logpdf(var_dist, value)
        sampler.variational_param_grads[addr] = logpdf_param_grads(var_dist, value)
        sampler.ELBO += logpdf(dist, value) - lpq
        return value
    end
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

            # reset gradients + update params
            params = Tracker.data.(params)
            params += rho .* grad
            variational_dists[addr] = update_params(var_dist, params)
        end
        
    end

    return UniversalMeanField(variational_dists)
end

export bbvi