module Graph
    import ProgressLogging: @progress
    import TinyPPL: Address, AbstractTraces, VIResult
    export Address

    include("../utils/utils.jl")
    include("core/core.jl")

    """
    Wrapper for the result of sample based inference algorithms, like MH or IS.
    Provides getters for retrieving all values / specific value of a given address.
    """
    struct GraphTraces <: AbstractTraces
        addresses_to_ix::Dict{Address,Int} # this do not have to be meaningful but often convienent
        data::Array{Float64}
        retvals::Vector{Any}
    end
    function GraphTraces(pgm::PGM, data::Array{Float64}, retvals::Vector{Any})
        addresses_to_ix = Dict{Any,Int}(pgm.addresses[i] => i for i in 1:pgm.n_latents)
        return GraphTraces(addresses_to_ix, data, retvals)
    end
    
    function Base.show(io::IO, traces::GraphTraces)
        print(io, "GraphTraces($(size(traces.data,2)) entries for $(size(traces.data,1)) nodes)")
    end
    
    retvals(traces::GraphTraces) = traces.retvals
    
    Base.length(traces::GraphTraces) = size(traces.data,2)

    function Base.getindex(traces::GraphTraces, node::Int)
        return traces.data[node, :]
    end
    
    function Base.getindex(traces::GraphTraces, node::Int, i::Int)
        return traces.data[node, i]
    end
    
    function subset(traces::GraphTraces, ixs)
        return GraphTraces(traces.addresses_to_ix, traces.data[:,ixs], traces.retvals[ixs])
    end
    export subset

    function Base.getindex(traces::GraphTraces, ::Colon, ix::Int)
        return traces.data[:,ix]
    end
    
    function Base.getindex(traces::GraphTraces, ::Colon, ixs)
        return subset(traces, ixs)
    end


    # use address getters only if it makes sense

    function Base.getindex(traces::GraphTraces, addr::Address)
        @assert haskey(traces.addresses_to_ix, addr) "$addr not in addresses_to_ix"
        return traces.data[traces.addresses_to_ix[addr], :]
    end
    
    function Base.getindex(traces::GraphTraces, addr::Address, i::Int)
        @assert haskey(traces.addresses_to_ix, addr) "$addr not in addresses_to_ix"
        return traces.data[traces.addresses_to_ix[addr], i]
    end
    
    export GraphTraces


    include("LW.jl")
    include("LMH.jl")
    include("logjoint.jl")
    include("HMC.jl")
    include("ADVI.jl")
    include("BBVI.jl")
    include("SMC.jl")
    include("compiled/LW.jl")
    include("compiled/LMH.jl")
    include("exact/exact.jl")


    import Random
    export Random
end