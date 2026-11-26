# Exact inference on the discrete graph-based models in models/, compared to their reference solutions.
#
# A discrete graph-based model is turned into a factor graph, on which
# - variable elimination (VE) sums out all variables that are not in the return expression,
# - belief propagation (BP) passes messages along the factor graph, which has to be a tree,
# - junction tree message passing (JT) passes messages between clusters of variables, which works for any factor graph.
# Each gives the exact distribution of the return value and the evidence (probability of the observations).
#
# Run all models:      julia --project=examples examples/exact_inference/run.jl
# Run selected models: julia --project=examples examples/exact_inference/run.jl burglary monty_hall

using TinyPPL.Distributions
using TinyPPL.Graph

# Distribution of the return value, given the result of an exact inference algorithm and the factor at its root.
# get_posterior(result) is the normalised posterior over the variables of this factor.
function return_distribution(model, result, root_factor)
    factor = FactorNode(root_factor.neighbours, log.(get_posterior(result)))
    distribution = Dict{Any,Float64}()
    for (value, p) in evaluate_return_expr_over_factor(model, factor)
        distribution[value] = get(distribution, value, 0.) + p
    end
    return sort(collect(distribution), by=first)
end

# prints the probabilities of return values with probability at least 0.001
function print_result(algorithm, model, result, root_factor)
    distribution = return_distribution(model, result, root_factor)
    println(rpad(algorithm, 4), join(["P($value) = $(round(p, digits=6))" for (value, p) in distribution if p >= 0.001], ", "),
        ", log evidence = ", round(result.log_evidence, digits=4))
end

model_names = isempty(ARGS) ? sort([splitext(file)[1] for file in readdir(joinpath(@__DIR__, "models"))]) : ARGS
for name in model_names
    # Each model file defines get_model() and print_reference_solution().
    # They are defined while this loop runs, so we call them with invokelatest.
    include(joinpath(@__DIR__, "models", name * ".jl"))
    model = Base.invokelatest(get_model)
    println("===== ", model.name, " =====")

    result = variable_elimination(model, order=:Greedy) # marginal over the variables of the return expression
    print_result("VE", model, result, result.factor)

    result = junction_tree_message_passing(model, return_factor_as_root=true)
    print_result("JT", model, result, result.root_factor)

    if is_tree(model)
        result = belief_propagation(model, return_factor_as_root=true)
        print_result("BP", model, result, result.root.node)
    else
        println("BP  not applicable, the factor graph is not a tree")
    end

    Base.invokelatest(print_reference_solution)
    println()
end
