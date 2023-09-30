

mutable struct StaticLW <: StaticSampler
    W::Float64
    trace::AbstractVector{Real}
    addresses_to_ix::Addr2Ix
    function StaticLW(addresses_to_ix::Addr2Ix)
        return new(0., Vector{Real}(), addresses_to_ix)
    end
end

function sample(sampler::StaticLW, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    value = rand(dist)
    sampler.trace[sampler.addresses_to_ix[addr]] = value
    
    return value
end

function likelihood_weighting(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int)
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(addresses)
    traces = Traces(addresses_to_ix, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = StaticLW(addresses_to_ix)
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = view(traces.data, :, i)
        retvals[i] = model(args, sampler, observations)
        logprobs[i] = sampler.W
    end
    return traces, retvals, normalise(logprobs)
end

function likelihood_weighting(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, trace::Addresses)
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(trace, addresses)
    @assert trace ⊆ addresses
    K = length(trace)
    traces = Traces(addresses_to_ix, K, n_samples)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = StaticLW(addresses_to_ix)
    sampler.trace = Vector{Real}(undef, length(addresses_to_ix))
    @progress for i in 1:n_samples
        sampler.W = 0.
        retvals[i] = model(args, sampler, observations)
        traces.data[:,i] = sampler.trace[1:K]
        logprobs[i] = sampler.W
    end
    return traces, retvals, normalise(logprobs)
end

export likelihood_weighting