
const AbstractUniversalTrace = Dict{Address, <:RVValue}
const UniversalTrace = Dict{Address, RVValue}

export AbstractUniversalTrace, UniversalTrace

"""
Wrapper for the result of sample based inference algorithms, like MH or IS.
Provides getters for retrieving all values / specific value of a given address.
"""
struct UniversalTraces <: AbstractTraces
    data::Vector{<:AbstractUniversalTrace}
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

function subset(traces::UniversalTraces, ixs)
    return UniversalTraces(traces.data[ixs], traces.retvals[ixs])
end
export subset

function Base.getindex(traces::UniversalTraces, ::Colon, ix::Int)
    return traces.data[ix]
end

function Base.getindex(traces::UniversalTraces, ::Colon, ixs)
    return subset(traces::UniversalTraces, ixs)
end


export UniversalTraces


include("Forward.jl")
include("logjoint.jl")
include("guide.jl")
include("LW.jl")
include("LMH.jl")
include("HMC.jl")
include("RWMH.jl")
include("ADVI.jl")
include("BBVI.jl")
include("IMCMC.jl")
include("trace_transform.jl")