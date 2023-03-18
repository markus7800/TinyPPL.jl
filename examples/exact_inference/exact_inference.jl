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

@info "Variable Elimination"
inference(true,algo=:VE)
print_reference_solution()

# b = @benchmark inference(algo=:VE)
# show(Base.stdout, MIME"text/plain"(), b)


@info "Junction Tree Message Passing"
inference(true,algo=:JT)
print_reference_solution()

if is_tree(model)
    @info "Belief Propagation"
    inference(true,algo=:BP)
    print_reference_solution()

    # b = @benchmark inference(algo=:BP)
    # show(Base.stdout, MIME"text/plain"(), b)
else
    @info "Cannot apply Belief Propagation"
end