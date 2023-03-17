using TinyPPL.Graph

println(ARGS[1])

include(ARGS[1]*".jl")

function inference(show_results=false; algo=:VE)
    model = get_model()

    if algo == :VE
        f = variable_elimination(model)
    else
        f, evidence = belief_propagation(model)
    end
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        println(algo)
        display(retvals)
    end
end

inference(true,algo=:VE)
# inference(true,algo=:BP)
print_reference_solution()

using BenchmarkTools
@info "Benchmark Variable Elimation"
b = @benchmark inference(algo=:VE)
show(Base.stdout, MIME"text/plain"(), b)
# @info "Benchmark Belief Propagation"
# b = @benchmark inference(algo=:BP)
# show(Base.stdout, MIME"text/plain"(), b)