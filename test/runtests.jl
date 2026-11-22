#=
Run all tests:            julia --project=. -e 'using Pkg; Pkg.test()'
Run selected test files:  julia --project=. -e 'using Pkg; Pkg.test(test_args=["graph", "exact"])'
Faster (no environment re-resolution, uses the package environment directly):
                          julia --project=. test/runtests.jl graph exact

Most tests compare inference results against reference posteriors (see references.jl).
Comments like `# C3` refer to the issue IDs in REVIEW.md.
=#
using Test
import Logging
Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Warn)) # silence @info / @progress

using TinyPPL

include("references.jl")
include("utils.jl")

const ALL_TESTS = ["distributions", "universal", "static", "graph", "exact"]
const SELECTED_TESTS = isempty(ARGS) ? ALL_TESTS : ARGS

@testset "TinyPPL" begin
    for name in SELECTED_TESTS
        @assert name in ALL_TESTS "Unknown test file $name. Choose from $ALL_TESTS."
        @testset "$name" begin
            include("test_$name.jl")
        end
    end
end
