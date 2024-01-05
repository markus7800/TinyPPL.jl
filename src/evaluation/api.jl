
const Address = Any
const RVValue = Real # currently only univariate distributions are supported
const Observations = Dict{Address, RVValue}
export Observations

const Trace = Dict{Address,RVValue}
"""
Wrapper for the result of sample based inference algorithms, like MH or IS.
Provides getters for retrieving all values / specific value of a given address.
"""
struct UniversalTraces
    data::Vector{Trace}
    retvals::Vector{Any}
end
retvals(traces::UniversalTraces) = traces.retvals
export retvals

function Base.show(io::IO, traces::UniversalTraces)
    print(io, "UniversalTraces($(length(traces.data)) entries)")
end

function Base.getindex(traces::UniversalTraces, addr::Address)
    return [get(t, addr, missing) for t in traces.data]
end
function Base.getindex(traces::UniversalTraces, addr::Address, i::Int)
    return get(traces.data[i], addr, missing)
end
Base.length(traces::UniversalTraces) = length(traces.data)
export UniversalTraces



abstract type VIResult end
function sample_posterior(::VIResult, n::Int)
    error("Not implemented.")
end
