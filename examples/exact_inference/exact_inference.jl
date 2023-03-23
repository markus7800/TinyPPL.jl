using TinyPPL.Graph
using BenchmarkTools

include(ARGS[1])

function inference(show_results=false; algo=:VE)
    model = get_model()

    if algo == :VE
        f = variable_elimination(model)
    elseif algo == :BP
        f, _ = belief_propagation(model)
    elseif algo == :JT
        f, _ = junction_tree_message_passing(model)
    end
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        display(retvals)
    end
end
model = get_model()
println(model.name)

@info "Variable Elimination"
inference(true,algo=:VE)
print_reference_solution()

b = @benchmark inference(algo=:VE)
show(Base.stdout, MIME"text/plain"(), b)
println()

# @info "Junction Tree Message Passing"
# inference(true,algo=:JT)
# print_reference_solution()

# b = @benchmark inference(algo=:JT)
# show(Base.stdout, MIME"text/plain"(), b)
# println()

if is_tree(model)
    @info "Belief Propagation"
    inference(true,algo=:BP)
    print_reference_solution()

    b = @benchmark inference(algo=:BP)
    show(Base.stdout, MIME"text/plain"(), b)
    println()
else
    @info "Cannot apply Belief Propagation"
end

function test_ve_order(model)
    @info "Test Variable Elimination Order"
    variable_nodes, factor_nodes = get_factor_graph(model)
    marginal_variables = return_expr_variables(model)
    for order in [:Topological, :MinNeighbours, :MinFill, :WeightedMinFill]
        @info order
        elimination_order = get_elimination_order(model, variable_nodes, marginal_variables, order)
        b = @benchmark variable_elimination($variable_nodes, $elimination_order)
        show(Base.stdout, MIME"text/plain"(), b)
        println()
    end
end

# test_ve_order(model)