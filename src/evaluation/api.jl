
abstract type AbstractTraces end

const AbstractUniversalTrace = Dict{Address, <:RVValue}
const UniversalTrace = Dict{Address, RVValue}

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
export UniversalTraces


"""
Wrapper for the result of variational inference methods like ADVI or BBVI.
Includes a method to produce samples from posterior, which returns UniversalTraces.
Usually wraps the variational distribution itself, where the parameters typically
are optimised to fit the unconstrained objective (i.e. directly sampling from the
wrapped distribution may not produce correct posterior samples - use `sample_posterior`
instead).
"""
abstract type VIResult end

# returns samples X and log Q(X)
function sample_posterior(::VIResult, n::Int)::Tuple{<:AbstractTraces,Vector{Float64}}
    error("Not implemented.")
end


import TinyPPL.Distributions: VariationalParameters
const Param2Ix = Dict{Address, UnitRange{Int}}

"""
Wrapper for VariationalParameters, which maps name to value.
"""
struct VIParameters
    phi::VariationalParameters
    params_to_ix::Param2Ix
end
function Base.show(io::IO, p::VIParameters)
    print(io, "VIParameters(")
    print(io, sort(collect(keys(p.params_to_ix)), lt = (x,y) -> first(p.params_to_ix[x]) < first(p.params_to_ix[y])))
    print(io, ")")
end
function Base.getindex(p::VIParameters, i::Int)
    return p.phi[i]
end
function Base.getindex(p::VIParameters, addr::Any)
    ix = p.params_to_ix[addr]
    if length(ix) == 1
        return p.phi[ix[1]]
    else
        return p.phi[ix]
    end
end