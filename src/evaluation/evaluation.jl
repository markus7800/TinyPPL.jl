module Evaluation
    import ..TinyPPL.Distributions: Distribution, logpdf
    import ProgressLogging: @progress
    import TinyPPL: Address, AbstractTraces, VIResult

    """
    Wrapper for VariationalParameters, which maps name to value.
    """
    abstract type VIParameters end
    function Base.getindex(p::VIParameters, addr::Address)
        error("Not implemented.")
    end

    include("../utils/utils.jl")
    include("sampler.jl")
    include("core.jl")
    # include("dependencies.jl")
    include("universal/universal.jl")
    include("static/static.jl")

    import Random
    export Random
end