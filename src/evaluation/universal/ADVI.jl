import Tracker
import Distributions
import ..Distributions: VariationalDistribution, VariationalNormal, get_params, update_params


mutable struct UniversalConstraintTransformer <: UniversalSampler
    X::Dict{Any,Float64}
    Y::Dict{Any,Float64}
    to::Symbol
    function UniversalConstraintTransformer(X::Dict{Any,Float64}, to::Symbol)
        @assert to in (:constrained, :unconstrained)
        return new(X, Dict{Any,Float64}(), to)
    end
end 
function sample(sampler::UniversalConstraintTransformer, addr::Any, dist::Distributions.DiscreteDistribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    sampler.Y[addr] = get(sampler.X, addr, mean(dist))
    return sampler.Y[addr]
end
function sample(sampler::UniversalConstraintTransformer, addr::Any, dist::Distributions.ContinuousDistribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    transformed_dist = to_unconstrained(dist)
    if sampler.to == :unconstrained
        constrained_value = get(sampler.X, addr, mean(dist))
        unconstrained_value = transformed_dist.T(constrained_value)
        sampler.Y[addr] = unconstrained_value
    else # samper.to == :constrained
        unconstrained_value = get(sampler.X, addr, 0.0)
        constrained_value = transformed_dist.T_inv(unconstrained_value)
        sampler.Y[addr] = constrained_value
    end
    return constrained_value
end

struct UniversalMeanField
    variational_dists::Dict{Any,VariationalDistribution}
end
function Base.getindex(umf::UniversalMeanField, addr)
    return umf.variational_dists[addr]
end

function Distributions.rand(umf::UniversalMeanField)
    X = Dict{Any,Float64}(addr => Distributions.rand(var_dist) for (addr, var_dist) in umf.variational_dists)
    return X
end

function Distributions.rand(umf::UniversalMeanField, n::Int)
    return [Distributions.rand(umf) for _ in 1:n]
end

function transform_to_constrained(X::Dict{Any,Float64}, model::UniversalModel, args::Tuple, observations::Dict)::Dict{Any,Float64}
    sampler = UniversalConstraintTransformer(X, :constrained)
    model(args, sampler, observations)
    return sampler.Y
end

function transform_to_constrained(Xs::Vector{Dict{Any,Float64}}, model::UniversalModel, args::Tuple, observations::Dict)::Vector{Dict{Any,Float64}}
    return [transform_to_constrained(X, model, args, observations) for X in Xs]
end

export transform_to_constrained


struct UniversalTraces
    data::Vector{Dict{Any,Real}}
end
function Base.getindex(traces::UniversalTraces, addr::Any)
    return [get(t, addr, missing) for t in traces.data]
end
export UniversalTraces

struct UniversalVIResult <: VIResult
    Q::Union{UniversalMeanField, Guide}
    transform_to_constrained::Function
end
function sample_posterior(res::UniversalVIResult, n::Int)
    samples = rand(res.Q, n)
    @assert samples isa Vector{Dict{Any,T}} where T <: Real
    samples = res.transform_to_constrained.(samples)
    return UniversalTraces(samples)
end
export sample_posterior

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
        q = VariationalNormal()
        params = Tracker.param.(get_params(q)) # no vector params
        sampler.variational_dists[addr] = update_params(q, params)
    end
    var_dist = sampler.variational_dists[addr]
    transformed_dist = to_unconstrained(dist)
    unconstrained_value = rand(var_dist)
    lpq = logpdf(var_dist, unconstrained_value)
    sampler.ELBO += logpdf(transformed_dist, unconstrained_value) - lpq
    constrained_value = transformed_dist.T_inv(unconstrained_value)

    return constrained_value
end

# Only Gaussian Mean Field
function advi_meanfield(model::UniversalModel, args::Tuple, observations::Dict,  n_samples::Int, L::Int, learning_rate::Float64)

    eps = 1e-8
    acc = Dict{Any, AbstractVector{Float64}}()
    pre = 1.1
    post = 0.9

    sampler = ADVI(Dict{Any, VariationalDistribution}())
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
            params = no_grad(params)
            params += rho .* grad
            sampler.variational_dists[addr] = update_params(var_dist, params)
        end
        
    end

    function _transform_to_constrained(X::Dict{Any,Float64})
        sampler = UniversalConstraintTransformer(X, :constrained)
        model(args, sampler, observations)
        return sampler.Y
    end
    return UniversalVIResult(UniversalMeanField(sampler.variational_dists), _transform_to_constrained)
end
export advi_meanfield

import ..Distributions: ELBOEstimator, estimate_elbo

function advi(model::UniversalModel, args::Tuple, observations::Dict, 
    n_samples::Int, L::Int, learning_rate::Float64,
    guide::UniversalModel, guide_args::Tuple, estimator::ELBOEstimator
    )

    logjoint = make_logjoint(model, args, observations)
    q = make_guide(guide, guide_args, Dict())

    # cannot use advi_logjoint because phi is not of constant size
    phi = no_grad(get_params(q))

    eps = 1e-8
    acc = fill(eps, size(phi))
    pre = 1.1
    post = 0.9

    @progress for _ in 1:n_samples
        # setup for gradient computation
        phi_tracked = Tracker.param(phi)
        q = update_params(q, phi_tracked)

        # estimate elbo
        elbo = estimate_elbo(estimator, logjoint, q, L)

        # automatically compute gradient
        Tracker.back!(elbo)
        phi_tracked = get_params(q) # still has grads because of growing with vcat
        phi = vcat(phi, zeros(length(phi_tracked) - length(phi)))

        grad = Tracker.grad(phi_tracked)

        # decayed adagrad update rule
        acc = vcat(acc, fill(eps, length(phi_tracked) - length(acc)))
        acc = @. post * acc + pre * grad^2
        rho = @. learning_rate / (sqrt(acc) + eps)
        phi += @. rho * grad
    end

    # guide has to propose in correct support,
    # if you want to fit guide to unconstrained model you have to do it manually and transform to constrained
    return UniversalVIResult(update_params(q, phi), identity)
end

export advi