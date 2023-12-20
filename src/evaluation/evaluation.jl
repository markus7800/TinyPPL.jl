module Evaluation
    import ..TinyPPL.Distributions: Distribution, logpdf
    import ProgressLogging: @progress

    abstract type VIResult end
    function sample_posterior(::VIResult, n::Int)
        error("Not implemented.")
    end

    include("../utils/utils.jl")
    include("sampler.jl")
    include("core.jl")
    include("dependencies.jl")
    include("universal/universal.jl")
    include("static/static.jl")
end