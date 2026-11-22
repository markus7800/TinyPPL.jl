import Tracker
import TinyPPL.Distributions: Normal, to_unconstrained

#=
Non-parametric HMC (Mak, Zaiser, Ong: Nonparametric Hamiltonian Monte Carlo, ICML 2021)

We work with unconstrained values (each continuous latent is mapped to ℝ with `to_unconstrained`).
The state of the sampler is a trace X: address → unconstrained value.

The number of addresses that the program uses may change with X.
To get a fixed-dimensional state, we consider an extended target over the union of all addresses
that we encountered, where addresses that are *not used* by the program are distributed
independently according to an auxiliary distribution q (here standard normal):

    π_ext(X) = p(X_used, Y) ∏_{addr unused} q(X[addr])
    U_ext(X) = -log π_ext(X)

The marginal of π_ext on the used addresses is the posterior, so we may discard unused addresses
after each iteration (and resample them from q when they are needed again).

(*) During leapfrog integration, the program may request an address that is not yet in X.
Conceptually, the value of this address was already part of the initial state (x0 ~ q, p0 ~ N(0,1)).
As the address was not used until step t, it did not influence the gradient, i.e. p stayed p0
and its value is x0 + t * eps * p0. We instantiate it accordingly and add its initial potential
-log q(x0) and kinetic energy p0^2/2 to the energy of the initial state.

Since the gradient only depends on positions, the leapfrog map is volume preserving and reversible
(with momentum flip), even though the set of used addresses changes along the trajectory.
=#

const NPHMCTrace = Dict{Address,Float64}

# auxiliary distribution for (unconstrained) values of addresses that are not used by the program
const NPHMC_AUX_DIST = Normal(0., 1.)

kinetic_energy(P::NPHMCTrace) = sum(p^2 for p in values(P); init=0.) / 2


"""
Evaluates log p(T^{-1}(X), Y) + log abs det ∇T^{-1}(X) for unconstrained trace X
and records the addresses that are used by the program.
"""
mutable struct NonParametricHMCLogjoint <: UniversalSampler
    W::Float64
    X::NPHMCTrace
    used::Set{Address}
    function NonParametricHMCLogjoint(X::NPHMCTrace)
        return new(0., X, Set{Address}())
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

    push!(sampler.used, addr)

    # map unconstrained value to support
    transformed_dist = to_unconstrained(dist)
    unconstrained_value = sampler.X[addr]
    sampler.W += logpdf(transformed_dist, unconstrained_value) # log p(T^{-1}(X)) + log abs det ∇T^{-1}(X)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

"""
Returns the potential energy of the extended target U_ext(X) = U_used + U_aux,
where U_used = -log p(X_used, Y) and U_aux = -∑_{addr unused} log q(X[addr]),
the return value of the program, and the set of used addresses.
"""
function get_U(model::UniversalModel, args::Tuple, observations::Observations, X::NPHMCTrace)
    sampler = NonParametricHMCLogjoint(X)
    retval = model(args, sampler, observations)
    U_used = -sampler.W
    U_aux = 0.
    for (addr, x) in X
        if !(addr in sampler.used)
            U_aux -= logpdf(NPHMC_AUX_DIST, x)
        end
    end
    return U_used, U_aux, retval, sampler.used
end


"""
Computes the log density of the program with tracked unconstrained values to get ∇U.
Instantiates addresses that are requested by the program, but are not in X yet.
"""
mutable struct NonParametricHMCGradExtend <: UniversalSampler
    W::Tracker.TrackedReal{Float64}
    X::Dict{Address,Tracker.TrackedReal{Float64}}
    drift_time::Float64            # t * eps, time that has passed since start of trajectory
    new_P::Dict{Address,Float64}   # initial momenta of instantiated addresses
    U_aux_initial::Float64         # -∑ log q(x0) of instantiated addresses
    function NonParametricHMCGradExtend(X::NPHMCTrace, drift_time::Float64)
        X_tracked = Dict{Address,Tracker.TrackedReal{Float64}}(addr => Tracker.param(x) for (addr, x) in X)
        return new(0., X_tracked, drift_time, Dict{Address,Float64}(), 0.)
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

    if !haskey(sampler.X, addr)
        # address was part of the initial state with x0 ~ q and p0 ~ N(0,1),
        # but did not influence the gradient so far -> moved with constant momentum p0, see (*) on top of the file
        x0 = rand(NPHMC_AUX_DIST)
        p0 = randn()
        sampler.new_P[addr] = p0
        sampler.U_aux_initial -= logpdf(NPHMC_AUX_DIST, x0)
        sampler.X[addr] = Tracker.param(x0 + sampler.drift_time * p0)
    end

    # map unconstrained value to support
    transformed_dist = to_unconstrained(dist)
    unconstrained_value = sampler.X[addr]
    sampler.W += logpdf(transformed_dist, unconstrained_value) # log p(T^{-1}(X)) + log abs det ∇T^{-1}(X)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

"""
Computes ∇U_used(X) and inserts addresses that are requested by the program into X.
Returns the gradient, the initial momenta of the inserted addresses and their initial auxiliary potential.
"""
function get_U_grad_and_extend!(model::UniversalModel, args::Tuple, observations::Observations, X::NPHMCTrace, drift_time::Float64)
    sampler = NonParametricHMCGradExtend(X, drift_time)
    model(args, sampler, observations)
    Tracker.back!(-sampler.W)

    grad = NPHMCTrace()
    for (addr, x) in sampler.X
        X[addr] = Tracker.data(x)
        grad[addr] = Tracker.grad(x) # 0 for addresses that are not used
    end
    return grad, sampler.new_P, sampler.U_aux_initial
end

"""
Leapfrog integration with L steps of size eps starting from X_current and P ~ N(0,I).
Returns the proposed trace and the kinetic energies of the initial and final state as well as
the auxiliary potential of addresses which were instantiated during integration.
All quantities refer to the same (extended) set of addresses.
"""
function universal_leapfrog(
    model::UniversalModel, args::Tuple, observations::Observations,
    X_current::NPHMCTrace,
    L::Int, eps::Float64
    )
    X = copy(X_current)
    P = NPHMCTrace(addr => randn() for addr in keys(X))
    K_current = kinetic_energy(P)
    U_aux_current = 0.

    for t in 0:L
        if t > 0
            # full step for position
            for (addr, p) in P
                X[addr] += eps * p
            end
        end

        grad, new_P, U_aux = get_U_grad_and_extend!(model, args, observations, X, t * eps)
        for (addr, p) in new_P
            P[addr] = p
        end
        K_current += kinetic_energy(new_P)
        U_aux_current += U_aux

        # half step for momentum at beginning and end, full step otherwise
        step_size = (t == 0 || t == L) ? eps / 2 : eps
        for (addr, g) in grad
            P[addr] -= step_size * g
        end
    end

    K_proposed = kinetic_energy(P)

    return X, U_aux_current, K_current, K_proposed
end

"""
Non-parametric HMC for universal models with continuous latent variables.
Returns a vector of traces with *unconstrained* values.
`x_initial` is a trace with constrained values (as returned by the model).
If no `x_initial` is given, the chain starts at a random prior sample (the first proposal is always accepted).
For models with discontinuities this sample may lie in a region that HMC cannot leave,
e.g. see examples/universal_non_parametric_hmc.jl. Then, it helps to initialise the chain with a likely trace
(for instance the trace with the largest weight among some likelihood weighting samples).
"""
function non_parametric_hmc(model::UniversalModel, args::Tuple, observations::Observations, n_samples::Int, L::Int, eps::Float64;
    x_initial::Union{Nothing,AbstractUniversalTrace}=nothing)

    # initialise x0
    if isnothing(x_initial)
        X_current = NPHMCTrace()
        U_current = Inf # first proposal is always accepted
        retval_current = nothing
    else
        X_unconstrained, _ = transform_to_unconstrained(x_initial, model, args, observations)
        X_current = NPHMCTrace(X_unconstrained)
        U_current, _, retval_current, used = get_U(model, args, observations, X_current)
        X_current = NPHMCTrace(addr => X_current[addr] for addr in used)
    end

    result = Vector{NPHMCTrace}(undef, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    n_accepted = 0
    @progress for i in 1:n_samples

        X_proposed, U_aux_current, K_current, K_proposed = universal_leapfrog(model, args, observations, X_current, L, eps)

        U_proposed, U_aux_proposed, retval_proposed, used = get_U(model, args, observations, X_proposed)

        # H_ext(current) = U_current + U_aux_current + K_current, where U_aux_current accounts for instantiated addresses
        # H_ext(proposed) = U_proposed + U_aux_proposed + K_proposed, where U_aux_proposed accounts for unused addresses
        if log(rand()) < (U_current + U_aux_current + K_current) - (U_proposed + U_aux_proposed + K_proposed)
            # discard unused addresses (marginalise auxiliary values)
            X_current = NPHMCTrace(addr => X_proposed[addr] for addr in used)
            U_current = U_proposed
            retval_current = retval_proposed
            n_accepted += 1
        end
        # Store regardless of acceptance
        result[i] = X_current
        retvals[i] = retval_current
    end

    @info "HMC" n_accepted/n_samples

    return result
end

export non_parametric_hmc
