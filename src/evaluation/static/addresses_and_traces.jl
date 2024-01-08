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

export AbstractUniversalTrace, UniversalTrace

"""
Wrapper for the result of sample based inference algorithms, like MH or IS.
Provides getters for retrieving all values / specific value of a given address.
"""
struct StaticTraces <: AbstractTraces
    addesses_to_ix::Addr2Ix
    data::Array{RVValue} # note: change here for multivariate support to UnitRange{Int}
    retvals::Vector{Any}
    function StaticTraces(addesses_to_ix::Addr2Ix, n::Int)
        return new(addesses_to_ix, Array{RVValue}(undef, length(addesses_to_ix), n), Vector{Any}(undef, n))
    end

    function StaticTraces(addesses_to_ix::Addr2Ix, k::Int, n::Int)
        return new(addesses_to_ix, Array{RVValue}(undef, k, n), Vector{Any}(undef, n))
    end
    
    function StaticTraces(addesses_to_ix::Addr2Ix, data::AbstractArray{<:RVValue}, retvals::Vector{Any})
        return new(addesses_to_ix, data, retvals)
    end
end

function Base.show(io::IO, traces::StaticTraces)
    print(io, "StaticTraces($(length(traces.retvals)) entries for $(length(traces.addesses_to_ix)) addresses)")
end

retvals(traces::StaticTraces) = traces.retvals

function Base.getindex(traces::StaticTraces, addr::Address)::Vector{<:RVValue}
    return traces.data[traces.addesses_to_ix[addr], :]
end

function Base.getindex(traces::StaticTraces, addr::Address, i::Int)::RVValue
    @assert addr in traces.addesses_to_ix
    return traces.data[traces.addesses_to_ix[addr], i]
end

# function Base.getindex(traces::Traces, addrs::Vector{Any}, i::Int)::Real
#     return traces.data[[traces.addesses_to_ix[addr] for addr in addrs], i]
# end

export StaticTraces