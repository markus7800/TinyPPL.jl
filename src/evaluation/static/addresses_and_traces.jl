import TinyPPL.Distributions: mean, mode

const Addr2Ix = Dict{Address, Int} # note: change here for multivariate support to UnitRange{Int}
const Addresses = Vector{Address} # addresses in order of execution

"""
Collects all addresses of sample statements (no observed value provided) by running the model.
Since static models instantiate the same set of addresses in each run, this only has to be done once.
The AddressCollector does not use `rand` but uses mean and mode to insert RV values. 
"""
struct AddressCollector <: StaticSampler
    addresses::Addresses
    function AddressCollector()
        return new(Addresses())
    end
end

function sample(sampler::AddressCollector, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end

    value = dist isa Distributions.DiscreteDistribution ? mode(dist) : mean(dist)
    push!(sampler.addresses, addr)
    
    return value
end

function get_addresses(model::StaticModel, args::Tuple, observations::Observations)::Addresses
    sampler = AddressCollector()
    model(args, sampler, observations)
    return sampler.addresses
end

function get_number_of_latent_variables(model::StaticModel, args::Tuple, observations::Observations)
    return length(get_addresses(model, args, observations))
end
export get_number_of_latent_variables


"""
Maps each address to an unique index.
"""
function get_address_to_ix(model::StaticModel, args::Tuple, observations::Dict)::Addr2Ix
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = Addr2Ix()
    for addr in addresses
        addresses_to_ix[addr] = length(addresses_to_ix)+1 # note: change here for multivariate support to UnitRange{Int}
    end
    return addresses_to_ix
end
export get_address_to_ix


const AbstractStaticTrace = AbstractVector{<:RVValue} # note: change here for multivariate support to UnitRange{Int}
const StaticTrace = Vector{Real}
StaticTrace(n::Int) = Vector{Real}(undef, n)

export AbstractStaticTrace, StaticTrace

"""
Wrapper for the result of sample based inference algorithms, like MH or IS.
Provides getters for retrieving all values / specific value of a given address.
"""
struct StaticTraces <: AbstractTraces
    addresses_to_ix::Addr2Ix
    data::Array{RVValue} # note: change here for multivariate support to UnitRange{Int}
    retvals::Vector{Any}
    function StaticTraces(addresses_to_ix::Addr2Ix, n::Int)
        return new(addresses_to_ix, Array{RVValue}(undef, length(addresses_to_ix), n), Vector{Any}(undef, n))
    end

    function StaticTraces(addresses_to_ix::Addr2Ix, k::Int, n::Int)
        return new(addresses_to_ix, Array{RVValue}(undef, k, n), Vector{Any}(undef, n))
    end
    
    function StaticTraces(addresses_to_ix::Addr2Ix, data::AbstractArray{<:RVValue}, retvals::Vector{Any})
        return new(addresses_to_ix, data, retvals)
    end
end

function Base.show(io::IO, traces::StaticTraces)
    print(io, "StaticTraces($(size(traces.data,2)) entries for $(length(traces.addresses_to_ix)) addresses)")
end

retvals(traces::StaticTraces) = traces.retvals

Base.length(traces::StaticTraces) = size(traces.data,2)

function Base.getindex(traces::StaticTraces, addr::Address)
    @assert haskey(traces.addresses_to_ix, addr) "$addr not in addresses_to_ix"
    return traces.data[traces.addresses_to_ix[addr], :]
end

function Base.getindex(traces::StaticTraces, addr::Address, i::Int)
    @assert haskey(traces.addresses_to_ix, addr) "$addr not in addresses_to_ix"
    return traces.data[traces.addresses_to_ix[addr], i]
end

function subset(traces::StaticTraces, ixs)
    return StaticTraces(traces.addresses_to_ix, traces.data[:,ixs], traces.retvals[ixs])
end
export subset

function Base.getindex(traces::StaticTraces, ::Colon, ix::Int)
    return traces.data[:,ix]
end

function Base.getindex(traces::StaticTraces, ::Colon, ixs)
    return subset(traces, ixs)
end

export StaticTraces