import Tracker
import Distributions
import TinyPPL.Distributions: VariationalDistribution, VariationalNormal, get_params, update_params

const ContinuousUniversalTrace = Dict{Address,Float64}

"""
Meanfield variational approximation. Maps address to variational distribution.
It is just a wrapper for ADVI / BBVI results and not <: VariationalDistribution
"""
struct UniversalMeanField
    variational_dists::Dict{Address,VariationalDistribution}
end
function Base.getindex(umf::UniversalMeanField, addr::Address)
    return umf.variational_dists[addr]
end

function rand_and_logpdf(umf::UniversalMeanField)
    X = ContinuousUniversalTrace()
    lp = 0.0
    for (addr, var_dist) in umf.variational_dists
        x, xlp = rand_and_logpdf(var_dist)
        lp += xlp
        X[addr] = x
    end
    return X, lp
end

# function rand_and_logpdf(umf::UniversalMeanField, n::Int)
#     Xs = Vector{ContinuousUniversalTrace}(undef, n)
#     lps = Vector{Float64}(undef, n)
#     for i in 1:n
#         @inbounds Xs[i], lps[i] = rand_and_logpdf(umf)
#     end
#     return Xs, lps
# end

# function Distributions.rand(umf::UniversalMeanField)
#     X = Dict{Any,Float64}(addr => Distributions.rand(var_dist) for (addr, var_dist) in umf.variational_dists)
#     return X
# end

# function Distributions.rand(umf::UniversalMeanField, n::Int)
#     return [Distributions.rand(umf) for _ in 1:n]
# end

"""
Wrapper for the result of universal ADVI and BBVI.
The only supported variational families are Meanfield and Guides.
Transforms values to constrained model in `sample_posterior` if required.
"""
struct UniversalVIResult <: VIResult
    Q::Union{UniversalMeanField, UniversalGuide}
    transform_to_constrained::Function # AbstractUniversalTrace -> Tuple{<:AbstractUniversalTrace}, Any}
end
function Base.show(io::IO, vi_result::UniversalVIResult)
    if vi_result.Q isa UniversalGuide
        print(io, "VIResult(UniversalGuide($(vi_result.Q.model.f)))")
    else
        print(io, "VIResult(UniversalMeanField($(length(vi_result.Q.variational_dists)) distributions))")
    end
end
function sample_posterior(res::UniversalVIResult, n::Int)
    Xs = Vector{ContinuousUniversalTrace}(undef, n)
    retvals = Vector{Any}(undef, n)
    lps = Vector{Float64}(undef, n)
    @progress for i in 1:n
        X, lp = rand_and_logpdf(res.Q)
        X, retval = res.transform_to_constrained(X)
        @inbounds Xs[i] = X
        @inbounds retvals[i] = retval
        @inbounds lps[i] = lp
    end
    return UniversalTraces(Xs, retvals), lps
end

mutable struct MeanFieldADVI <: UniversalSampler
    ELBO::Tracker.TrackedReal{Float64}
    variational_dists::Dict{Address,VariationalDistribution}

    function MeanFieldADVI(variational_dists::Dict{Address,VariationalDistribution})
        return new(0., variational_dists)
    end
end
function sample(sampler::MeanFieldADVI, addr::Address, dist::Distributions.DiscreteDistribution, obs::Nothing)::RVValue
    error("Discrete sample encountered in ADVI: $addr ~ $dist")
end

function sample(sampler::MeanFieldADVI, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.ELBO += logpdf(dist, obs)
        return obs
    end

    if !haskey(sampler.variational_dists, addr)
        # Gaussian approximation for every variable
        q = VariationalNormal()
        params::Tracker.TrackedVector{Float64, Vector{Float64}} = Tracker.param(get_params(q))
        sampler.variational_dists[addr] = update_params(q, params)
    end

    # fit Gaussian to unconstrained model
    var_dist = sampler.variational_dists[addr]
    transformed_dist = to_unconstrained(dist)
    unconstrained_value = rand(var_dist)
    lpq = logpdf(var_dist, unconstrained_value)
    sampler.ELBO += logpdf(transformed_dist, unconstrained_value) - lpq
    constrained_value = transformed_dist.T_inv(unconstrained_value)

    return constrained_value
end


"""
ADVI with Gaussian Meanfield approximation, where we fit unconstrained model.

Alternative implementation: Use MostSpecificDict{VariationalDistribution} and copy at new address,
since we cannot share VariationalDistribution among addresses.
If no VariationalDistribution is found at address, fit Gaussian to unconstrained distribution.

If you want this alternative behavior use ADVI with a guide program.

ELBO is optimised with automatic differentiation (AD).
"""
function advi_meanfield(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, learning_rate::Float64)

    eps = 1e-8
    acc = Dict{Address, AbstractVector{Float64}}()
    pre = 1.1
    post = 0.9

    sampler = MeanFieldADVI(Dict{Address, VariationalDistribution}())
    local params::Vector{Float64}
    local tracked_params::Tracker.TrackedVector{Float64, Vector{Float64}}
    @progress for i in 1:n_samples

        for (addr, var_dist) in sampler.variational_dists
            params = get_params(var_dist)
            @assert !Tracker.istracked(params)
            tracked_params = Tracker.param(params)
            sampler.variational_dists[addr] = update_params(var_dist, tracked_params)
        end

        # MonteCarlo estimation of ELBO gradient
        sampler.ELBO = 0.
        for _ in 1:L
            _ = model(args, sampler, observations)
        end
        sampler.ELBO = sampler.ELBO / L
        Tracker.back!(sampler.ELBO)

        # adagrad update for each address
        for (addr, var_dist) in sampler.variational_dists
            tracked_params = get_params(var_dist)
            grad = Tracker.grad(tracked_params)

            acc_addr = get(acc, addr, fill(eps,size(grad)))
            acc_addr = post .* acc_addr .+ pre .* grad.^2
            acc[addr] = acc_addr
            rho = learning_rate ./ (sqrt.(acc_addr) .+ eps)

            # reset gradients + update params
            params = no_grad(tracked_params)
            params += rho .* grad
            sampler.variational_dists[addr] = update_params(var_dist, params)
        end
        
    end

    _transform_to_constrained(X::AbstractUniversalTrace) = transform_to_constrained(X, model, args, observations)
    return UniversalVIResult(UniversalMeanField(sampler.variational_dists), _transform_to_constrained)
end
export advi_meanfield

import TinyPPL.Distributions: ELBOEstimator, estimate_elbo


"""
ADVI with variational distributions given by guide program.
Guide has to provide values in the correct support (absolute continuity).
Guide is fitted by default to original model, but can also be fitted to unconstrained model,
by setting `unconstrained = true`.
ELBO is optimised with automatic differentiation (AD).

This is almost the same as advi_logjoint, but the size of phi is not constant.
"""
function advi(model::UniversalModel, args::Tuple, observations::Observations, 
    n_samples::Int, L::Int, learning_rate::Float64,
    guide::UniversalModel, guide_args::Tuple, estimator::ELBOEstimator;
    unconstrained::Bool=false)

    if unconstrained
        logjoint = make_unconstrained_logjoint(model, args, observations)
        _transform_to_constrained(X::AbstractUniversalTrace) = transform_to_constrained(X, model, args, observations)
        _viresult_map = _transform_to_constrained
    else
        logjoint = make_logjoint(model, args, observations)
        _no_transform(X::AbstractUniversalTrace) = X, model(args, TraceSampler(X) , observations)
        _viresult_map = _no_transform
    end

    q = make_guide(guide, guide_args)

    # cannot use advi_logjoint because phi is not of constant size
    phi = no_grad(get_params(q))

    eps = 1e-8
    acc = Dict{Any, AbstractVector{Float64}}()
    pre = 1.1
    post = 0.9

    @progress for _ in 1:n_samples
        # setup for gradient computation
        phi_tracked = Dict{Address,Tracker.TrackedVector{Float64, Vector{Float64}}}(
            addr => Tracker.param(param) for (addr, param) in phi
        )
        q = update_params(q, phi_tracked)

        # estimate elbo
        elbo = estimate_elbo(estimator, logjoint, q, L)

        # automatically compute gradient
        Tracker.back!(elbo)
        phi_tracked = get_params(q) # still has grads because of growing with vcat

        for (addr, params) in phi_tracked
            grad = Tracker.grad(params) # defaults to zeros

            # adagrad update
            acc_addr = get(acc, addr, fill(eps,size(grad)))
            acc_addr = post .* acc_addr .+ pre .* grad.^2
            acc[addr] = acc_addr
            rho = learning_rate ./ (sqrt.(acc_addr) .+ eps)

            phi[addr] = get(phi, addr, grad isa Float64 ? 0. : zeros(size(grad))) + rho .* grad
        end
    end

    return UniversalVIResult(update_params(q, phi), _viresult_map)
end

export advi