const Address = Any
export Address

"""
Wrapper for the result of sample based inference algorithms, like MH or IS.
Provides getters for retrieving all values / specific value of a given address.
"""
abstract type AbstractTraces end
function Base.getindex(::AbstractTraces, ::Address)
    error("Not implemented!")
end
function Base.getindex(::AbstractTraces, ::Address, ::Int)
    error("Not implemented!")
end
function retvals(::AbstractTraces)
    error("Not implemented!")
end

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