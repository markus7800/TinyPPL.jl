module Evaluation
    import ..TinyPPL.Distributions: Distribution, logpdf
    import ProgressLogging: @progress
    import TinyPPL: Address, AbstractTraces, VIResult, VIParameters
    export Address

    include("../utils/utils.jl")
    include("sampler.jl")
    include("core.jl")
    # include("dependencies.jl")
    include("universal/universal.jl")
    include("static/static.jl")

    import Random
    export Random
end