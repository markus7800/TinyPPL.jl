module Evaluation
    import ..TinyPPL.Distributions: Distribution, logpdf
    import ProgressLogging: @progress

    include("../utils/utils.jl")
    include("sampler.jl")
    include("core.jl")
    include("dependencies.jl")
    include("variational.jl")
    include("universal/universal.jl")
    include("static/static.jl")
end