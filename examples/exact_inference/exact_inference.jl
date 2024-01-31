using TinyPPL.Graph
# using BenchmarkTools

# julia --project=. examples/exact_inference/exact_inference.jl burglary
include(ARGS[1]*".jl")

const BENCHMARK = false

function inference(model, show_results=false; algo=:VE, kwargs...)
    # model = get_model() TODO: this does not work for ladder, check why, we have to invoke from Main?
    
    if algo == :VE
        f, _ = greedy_variable_elimination(model; kwargs...)
    elseif algo == :BP || algo == :JT
        func = algo == :BP ? belief_propagation : junction_tree_message_passing
        t = func(model; kwargs...)
        f = t[1]
        if show_results && length(t) == 3
            marginals = t[3]
            for (_, address, table) in marginals
                println(address, ": ", table)
            end
        end
    end
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        display(retvals)
    end
end
model = get_model()
println(model.name)
# println(model)

@info "Variable Elimination"
inference(model,true,algo=:VE)
print_reference_solution()

# if BENCHMARK
#     b = @benchmark inference(model,algo=:VE)
#     show(Base.stdout, MIME"text/plain"(), b)
# end

println()

@info "Junction Tree Message Passing"
inference(model,true,algo=:JT)
print_reference_solution()

# if BENCHMARK
#     b = @benchmark inference(model,algo=:JT)
#     show(Base.stdout, MIME"text/plain"(), b)
# end

println()

if is_tree(model)
    @info "Belief Propagation"
    if model.name == :Survey
        inference(model,true,algo=:BP,all_marginals=true)
    else
        inference(model,true,algo=:BP)
    end
    print_reference_solution()

    # if BENCHMARK
    #     b = @benchmark inference(model,algo=:BP)
    #     show(Base.stdout, MIME"text/plain"(), b)
    #     println()

    #     println("All marginals")
    #     b = @benchmark inference(model,algo=:BP,all_marginals=true)
    #     show(Base.stdout, MIME"text/plain"(), b)
    # end

    println()
else
    @info "Cannot apply Belief Propagation"
end

# function test_ve_order(model)
#     @info "Test Variable Elimination Order"
#     variable_nodes, factor_nodes = get_factor_graph(model)
#     marginal_variables = return_expr_variables(model)
#     for order in [:Topological, :MinNeighbours, :MinFill, :WeightedMinFill]
#         @info order
#         elimination_order = get_elimination_order(model, variable_nodes, marginal_variables, order)
#         b = @benchmark variable_elimination($variable_nodes, $elimination_order)
#         show(Base.stdout, MIME"text/plain"(), b)
#         println()
#     end
# end

# test_ve_order(model)