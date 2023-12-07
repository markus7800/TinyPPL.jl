import Tracker
import Distributions

mutable struct ADVI <: UniversalSampler
    ELBO::Tracker.TrackedReal{Float64}
    variational_dists::Dict{Any, VariationalDistribution}

    function ADVI(variational_dists)
        return new(0., variational_dists)
    end
end

function sample(sampler::ADVI, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.ELBO += logpdf(dist, obs)
        return obs
    end

    if !haskey(sampler.variational_dists, addr)
        q = VariationalDistribution(dist)
        params = Tracker.param.(initial_params(q)) # no vector params
        sampler.variational_dists[addr] = update_params(q, params)
    end
    var_dist = sampler.variational_dists[addr]
    value, lpq = rand_and_logpdf(var_dist)
    sampler.ELBO += logpdf(dist, value) - lpq

    return value
end

function advi(model::UniversalModel, args::Tuple, observations::Dict,  n_samples::Int, L::Int, learning_rate::Float64, variational_dists::Dict{Any, VariationalDistribution})

    eps = 1e-8
    acc = Dict{Any, AbstractVector{Float64}}()
    pre = 1.1
    post = 0.9

    sampler = ADVI(variational_dists)
    @progress for i in 1:n_samples

        for (addr, var_dist) in sampler.variational_dists
            params = get_params(var_dist)
            @assert !any(Tracker.istracked.(params))
            params = Tracker.param.(params) # no vector params
            sampler.variational_dists[addr] = update_params(var_dist, params)
        end

        sampler.ELBO = 0.
        for _ in 1:L
            _ = model(args, sampler, observations)
        end
        sampler.ELBO = sampler.ELBO / L
        Tracker.back!(sampler.ELBO)

        for (addr, var_dist) in sampler.variational_dists
            params = get_params(var_dist)
            grad = Tracker.grad.(params)
            acc_addr = get(acc, addr, fill(eps,size(grad)))
            acc_addr = post .* acc_addr .+ pre .* grad.^2
            acc[addr] = acc_addr
            rho = learning_rate ./ (sqrt.(acc_addr) .+ eps)

            # reset gradients + update params
            params = Tracker.data.(params)
            params += rho .* grad
            sampler.variational_dists[addr] = update_params(var_dist, params)
        end
        
    end

    return sampler.variational_dists
end

export advi