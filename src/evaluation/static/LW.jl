
"""
Samples values at sample statement and stores them in trace if no observed value is provided,
else uses the observed value and accumulates its log density in `W`.
"""
mutable struct StaticLW <: StaticSampler
    W::Float64 # log p(Y|X)
    trace::AbstractStaticTrace
    addresses_to_ix::Addr2Ix
    function StaticLW(addresses_to_ix::Addr2Ix)
        return new(0., StaticTrace(), addresses_to_ix)
    end
end

function sample(sampler::StaticLW, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end

    value = rand(dist)
    sampler.trace[sampler.addresses_to_ix[addr]] = value
    
    return value
end

"""
Likelihood weighting is a special case of importance sampling where the reference distribution is
the program prior p(X).
See evaluation/universal/LW.jl
"""
function likelihood_weighting(model::StaticModel, args::Tuple, observations::Observations, n_samples::Int)
    addresses_to_ix = get_address_to_ix(model, args, observations)
    traces = StaticTraces(addresses_to_ix, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = StaticLW(addresses_to_ix)
    @progress for i in 1:n_samples
        sampler.W = 0.
        sampler.trace = view(traces.data, :, i)
        traces.retvals[i] = model(args, sampler, observations)
        logprobs[i] = sampler.W
    end
    return traces, normalise(logprobs)
end



"""
Sets up addresses_to_ix such that address in `first` are index before other addresses in `all`.

"""
function get_address_to_ix(first::Addresses, all::Addresses)::Addr2Ix
    @assert first ⊆ all
    addresses_to_ix = Addr2Ix()
    for addr in first
        addresses_to_ix[addr] = length(addresses_to_ix)+1
    end
    for addr in setdiff(all, first)
        addresses_to_ix[addr] = length(addresses_to_ix)+1
    end
    return addresses_to_ix
end

"""
Same as likelihood_weighting but only stores values of RVs with address in `trace`.
"""
function likelihood_weighting(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, trace::Addresses)
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(trace, addresses)
    K = length(trace)
    traces = StaticTraces(Addr2Ix(addr => ix for (addr, ix) in addresses_to_ix if addr in trace), K, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    sampler = StaticLW(addresses_to_ix)
    sampler.trace = StaticTrace(length(addresses_to_ix))
    @progress for i in 1:n_samples
        sampler.W = 0.
        traces.retvals[i] = model(args, sampler, observations)
        traces.data[:,i] = sampler.trace[1:K]
        logprobs[i] = sampler.W
    end
    return traces, normalise(logprobs)
end

export likelihood_weighting