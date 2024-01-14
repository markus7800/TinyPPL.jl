import Tracker
using Dictionaries

mutable struct NonParametricHMCLogjoint <: UniversalSampler
    W::Float64
    X::Dictionary{Address,Float64}
    seen::Union{Nothing,Dictionary{Address,Bool}}
    function NonParametricHMCLogjoint(X::Dictionary{Address,Float64}, seen::Union{Nothing,Dictionary{Address,Bool}}=nothing)
        return new(0., X, seen)
    end
end

function sample(sampler::NonParametricHMCLogjoint, addr::Address, dist::Distributions.DiscreteDistribution, obs::RVValue)::RVValue
    sampler.W += logpdf(dist, obs)
    return obs
end

function sample(sampler::NonParametricHMCLogjoint, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    if !isnothing(sampler.seen)
        sampler.seen[addr] = true # insert or update
    end

    # map unconstrained value to support
    transformed_dist = to_unconstrained(dist)
    unconstrained_value = sampler.X[addr]
    sampler.W += logpdf(transformed_dist, unconstrained_value) # log p(T^{-1}(X)) + log abs det ∇T^{-1}(X)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

mutable struct NonParametricHMCGradExtend <: UniversalSampler
    W::Tracker.TrackedReal{Float64}
    X::Dictionary{Address,Tracker.TrackedReal{Float64}}
    function NonParametricHMCGradExtend(X::Dictionary{Address,Tracker.TrackedReal{Float64}})
        return new(0., X)
    end
end

function sample(sampler::NonParametricHMCGradExtend, addr::Address, dist::Distributions.DiscreteDistribution, obs::RVValue)::RVValue
    sampler.W += logpdf(dist, obs)
    return obs
end

function sample(sampler::NonParametricHMCGradExtend, addr::Address, dist::Distributions.ContinuousDistribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    transformed_dist = to_unconstrained(dist)

    if !haskey(sampler.X, addr)
        # if this happens during leapfrog step t ∈ [1,...,L], then we know that
        # ∂U / ∂X_i[addr] = 0 for all i < t. Thus P_i is constant up until t.
        # Thus X_i gets updated with P_i t times, in total P_i * t * eps
        # As we can set the inital value arbitrarily, we sample P_i ~ Normal(0,1)
        # and "set" X_0[addr] = X_i[addr] - P_i * t * eps
        insert!(sampler.X, addr, Tracker.param(rand(transformed_dist)))
    end

    # map unconstrained value to support
    unconstrained_value = sampler.X[addr]
    sampler.W += logpdf(transformed_dist, unconstrained_value) # log p(T^{-1}(X)) + log abs det ∇T^{-1}(X)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end


function get_U(model::UniversalModel, args::Tuple, observations::Observations, X::Dictionary{Address,Float64}, seen::Union{Nothing,Dictionary{Address,Bool}})
    sampler = NonParametricHMCLogjoint(X, seen)
    model(args, sampler, observations)
    return -sampler.W
end

function get_U_grad_and_extend!(model::UniversalModel, args::Tuple, observations::Observations, X::Dictionary{Address,Float64}, P::Dictionary{Address,Float64})
    sampler = NonParametricHMCGradExtend(Tracker.param.(X)) # index is shared between X and X_tracked
    model(args, sampler, observations)
    Tracker.back!(-sampler.W)
    grad = Tracker.grad.(sampler.X)

    K = 0.

    for addr in setdiff(keys(sampler.X), keys(P)) # optimise
        p = randn()
        K += p^2
        insert!(P, addr, p) # initial value provided here
        #X[addr] = Tracker.data(sampler.X[addr])
        #set!(X, addr, Tracker.data(sampler.X[addr])) # initial value provided by constant
    end
    X = Tracker.data.(sampler.X)

    return X, grad, K
end

function universal_leapfrog(
    model::UniversalModel, args::Tuple, observations::Observations,
    X_current::Dictionary{Address,Float64},
    L::Int, eps::Float64
    )
    X = Dictionary{Address,Float64}(copy(X_current.indices), copy(X_current.values))
    P = Dictionary{Address,Float64}(keys(X), randn(length(X)))

    K_current = sum(P.^2)
    
    X, grad, K = get_U_grad_and_extend!(model, args, observations, X, P)
    K_current += K

    P = P .- eps/2 .* grad
    for _ in 1:(L-1)
        X = X .+ eps .* P

        X, grad, K = get_U_grad_and_extend!(model, args, observations, X, P)
        K_current += K

        P = P .- eps .* grad
    end
    X = X .+ eps .* P

    X, grad, K = get_U_grad_and_extend!(model, args, observations, X, P)
    K_current += K

    P = P .- eps/2 .* grad
    
    K_proposed = sum(P .^ 2)

    return X, K_current, K_proposed
end

function non_parametric_hmc(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, eps::Float64;
    x_initial::Union{Nothing,AbstractUniversalTrace}=nothing)

    # initialise x0
    X_current = isnothing(x_initial) ? Dictionary{Address,Float64}() : todo(x_initial)
    U_current = Inf

    result = Vector{Any}(undef, n_samples)
    n_accepted = 0
    @progress for i in 1:n_samples

        X_proposed, K_current, K_proposed = universal_leapfrog(model, args, observations, X_current, L, eps)

        # Compute new kinetic and potential energy
        seen = Dictionary{Address,Bool}(keys(X_proposed), falses(length(X_proposed)))
        U_proposed = get_U(model, args, observations, X_proposed, seen)

        # With perfect precision the leapfrog integrator should preserve the energy and accept with probability 1.
        # But it is an approximation and we adjust with a metropolis hasting step
        if log(rand()) < (U_current - U_proposed + K_current - K_proposed)

            # reduce X
            for (addr, seen) in pairs(seen) # optimise
                if !seen
                    delete!(X, addr)
                end
            end

            U_current = U_proposed
            X_current = X_proposed
            n_accepted += 1
        end
        # Store regardless of acceptance
        result[i] = X_current
    end

    @info "HMC" n_accepted/n_samples

    return result
end

export non_parametric_hmc