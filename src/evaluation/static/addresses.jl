
const Addr2Ix = Dict{Any, Int}
const Addresses = Set{Any}

struct AddressCollector <: StaticSampler
    addresses::Addresses
    function AddressCollector()
        return new(Addresses())
    end
end

function sample(sampler::AddressCollector, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end

    value = rand(dist)
    push!(sampler.addresses, addr)
    
    return value
end

function get_addresses(model::StaticModel, args::Tuple, observations::Dict)::Addresses
    sampler = AddressCollector()
    model(args, sampler, observations)
    return sampler.addresses
end

function get_address_to_ix(first::Addresses, second::Addresses=Addresses())::Addr2Ix
    addresses_to_ix = Addr2Ix()
    for addr in first
        addresses_to_ix[addr] = length(addresses_to_ix)+1
    end
    for addr in setdiff(second, first)
        addresses_to_ix[addr] = length(addresses_to_ix)+1
    end
    return addresses_to_ix
end

function get_address_to_ix(model::StaticModel, args::Tuple, observations::Dict)::Addr2Ix
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(addresses)
    return addresses_to_ix
end
export get_address_to_ix

struct Traces
    addesses_to_ix::Addr2Ix
    data::Array{Real}
    function Traces(addesses_to_ix::Addr2Ix, n::Int)
        return new(addesses_to_ix, Array{Real}(undef, length(addesses_to_ix), n))
    end

    function Traces(addesses_to_ix::Addr2Ix, k::Int, n::Int)
        return new(addesses_to_ix, Array{Real}(undef, k, n))
    end
    
    function Traces(addesses_to_ix::Addr2Ix, data::AbstractArray{<:Real})
        return new(addesses_to_ix, data)
    end
end

function Base.getindex(traces::Traces, addr::Any)::Vector{Real}
    return traces.data[traces.addesses_to_ix[addr], :]
end

function Base.getindex(traces::Traces, addr::Any, i::Int)::Real
    @assert addr in traces.addesses_to_ix
    return traces.data[traces.addesses_to_ix[addr], i]
end

function Base.getindex(traces::Traces, addrs::Vector{Any}, i::Int)::Real
    return traces.data[[traces.addesses_to_ix[addr] for addr in addrs], i]
end

export Traces