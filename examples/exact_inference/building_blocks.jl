# The building blocks of exact inference for discrete graph-based models (see run.jl for running the algorithms).
#
# - Factors are tables over the joint support of some variables (in log space), with product, sum and division.
# - Variable elimination sums out variables one by one; the elimination order determines the size of intermediate factors.
# - A junction tree groups variables into clusters. After message passing in both directions (calibration),
#   it answers queries about any set of variables and can draw exact samples from the posterior.
# References: Koller, Friedman: Probabilistic Graphical Models (2009), Chapters 9 and 10;
# Holtzen, Van den Broeck, Millstein: Scaling Exact Inference for Discrete Probabilistic Programs (OOPSLA 2020),
# whose diamond and ladder networks we use to test scaling.
#
# Run: julia --project=examples examples/exact_inference/building_blocks.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import TinyPPL: logsumexp
import Random

# ===== Factor operations =====
# Factors over X ∈ {1, 2}, Y ∈ {1, 2, 3}, Z ∈ {1, 2, 3, 4}. Tables hold log values.
Random.seed!(0)
X = VariableNode(1, :X); X.support = [1., 2.]
Y = VariableNode(2, :Y); Y.support = [1., 2., 3.]
Z = VariableNode(3, :Z); Z.support = [1., 2., 3., 4.]
A = FactorNode([X, Y], log.(rand(2, 3)))
B = FactorNode([Y, Z], log.(rand(3, 4)))

# The product of A(X, Y) and B(Y, Z) is C(X, Y, Z) = A(X, Y) B(Y, Z).
C = factor_product(A, B)
println("C = A * B: ", C, ", C(x, y, z) = A(x, y) B(y, z): ",
    all(C.table[x, y, z] ≈ A.table[x, y] + B.table[y, z] for x in 1:2, y in 1:3, z in 1:4))

# Summing out Z gives a factor over X and Y.
println("sum over Z: ", factor_sum(C, [Z]), ", equals A(x, y) ∑_z B(y, z): ",
    factor_sum(C, [Z]).table ≈ [A.table[x, y] + logsumexp(B.table[y, :]) for x in 1:2, y in 1:3])

# factor_division!(C, A, out) writes C / A into out. Its scope can be the scope of B or of C (then B is broadcast).
println("C / A recovers B: ", factor_division!(C, A, FactorNode(B.neighbours, similar(B.table))).table ≈ B.table,
    ", C / B recovers A: ", factor_division!(C, B, FactorNode(A.neighbours, similar(A.table))).table ≈ A.table)

# Zero entries (log value -Inf) are allowed. Dividing by zero is only needed where the numerator is also zero,
# and gives 0 by convention, as in belief propagation with division.
A0 = FactorNode([X, Y], log.([0. 1. 1.; 0. 0. 1.]))
B0 = FactorNode([Y, Z], log.(Float64[0 1 1 0; 1 0 1 1; 1 1 0 0]))
C0 = factor_product(A0, B0)
println("with zeros: C / A recovers B where A > 0: ",
    all(factor_division!(C0, A0, FactorNode(C0.neighbours, similar(C0.table))).table[x, y, z] ≈ B0.table[y, z]
        for x in 1:2, y in 1:3, z in 1:4 if isfinite(A0.table[x, y])))


# ===== Variable elimination and elimination orders =====
# A chain of N noisy network routers ("diamond" network, Holtzen et al.): each router forwards the packet
# along one of two routes, one of which drops it with probability 0.001. P(packet arrives) = (1 - 0.001/2)^N.
N = 100
model = @pgm Diamond begin
    function or(x, y)
        max(x, y)
    end
    function and(x, y)
        min(x, y)
    end
    function diamond(s1)
        let route ~ Bernoulli(0.5),
            s2 = route == 1. ? s1 : 0.,
            s3 = route == 1. ? 0. : s1,
            drop ~ Bernoulli(0.001)

            or(s2, and(s3, 1 - drop))
        end
    end
    function step(old_net)
        let net ~ Dirac(diamond(old_net))
            net
        end
    end
    @iterate($(N), step, 1.)
end

println("\ndiamond network with N = $N: P(packet arrives) = ", (1 - 0.001 / 2)^N)
# Finding a good elimination order is NP-hard. The heuristics are applied before or during elimination (:Greedy).
for order in [:Topological, :MinNeighbours, :MinFill, :WeightedMinFill, :Greedy]
    variable_elimination(model, order=order) # compile
    t = @elapsed result = variable_elimination(model, order=order)
    p = Dict(evaluate_return_expr_over_factor(model, result.factor))[1.]
    println("  VE with order ", rpad(order, 16), round(t, digits=3), "s: P(packet arrives) = ", p)
end


# ===== Scaling to large networks =====
# models/diamond.jl and models/ladder.jl construct the factor graphs of these networks directly, for N = 5000.
# Variable elimination and junction trees scale linearly in N for these networks.
N = 5000
for name in ["diamond", "ladder"]
    include(joinpath(@__DIR__, "models", name * ".jl"))
    variable_nodes, factor_nodes, marginal_variables, return_factor = Base.invokelatest(get_model_factor_graph, N)
    println("\n", name, " network with N = $N: ", length(variable_nodes), " variables")

    t = @elapsed begin
        order = get_greedy_elimination_order(variable_nodes, marginal_variables)
        result = variable_elimination(variable_nodes, order)
    end
    println("  VE: ", round(t, digits=2), "s, posterior of the return variables ", round.(get_posterior(result), digits=6))

    # belief propagation needs a factor graph without cycles
    if is_tree(variable_nodes, factor_nodes)
        t = @elapsed result = belief_propagation(get_belief_tree(return_factor), return_factor, false)
        println("  BP: ", round(t, digits=2), "s, posterior of the return variables ", round.(get_posterior(result), digits=6))
    else
        println("  BP: not applicable, the factor graph is not a tree")
    end

    t = @elapsed begin
        order = get_greedy_elimination_order(variable_nodes, Int[])
        junction_tree, root, root_factor = get_junction_tree(variable_nodes, order, return_factor)
        result = junction_tree_message_passing(junction_tree, root, root_factor, false)
    end
    println("  JT: ", round(t, digits=2), "s, posterior of the return variables ", round.(get_posterior(result), digits=6))
    print("  "); Base.invokelatest(print_reference_solution, N)
end


# ===== Junction trees =====
# The student network (Koller, Friedman, Figure 9.8) with uniform distributions, as we only care about its structure.
model = @pgm Student begin
    let C ~ Categorical([1.]),
        D ~ Categorical(C == 1. ? 1. : 1.),
        I ~ Categorical([1.]),
        G ~ Categorical(D == 1. && I == 1. ? 1. : 1.),
        L ~ Categorical(G == 1. ? 1. : 1.),
        S ~ Categorical(I == 1. ? 1. : 1.),
        J ~ Categorical(L == 1. && S == 1. ? 1. : 1.),
        H ~ Categorical(G == 1. && J == 1. ? 1. : 1.)

        J
    end
end

# Eliminating variables in the order C, D, I, H, G, S, L, J gives the clusters of Koller, Friedman, Figure 10.6.
variable_nodes, factor_nodes = get_factor_graph(model)
return_factor = add_return_factor!(model, variable_nodes, factor_nodes)
variable(name) = only(v for v in variable_nodes if v.address == name)
elimination_order = variable.([:C, :D, :I, :H, :G, :S, :L, :J])
junction_tree, root, root_factor = get_junction_tree(variable_nodes, elimination_order, return_factor)
println("\njunction tree of the student network:")
print_junction_tree(root)


# ===== Queries on a calibrated junction tree =====
# A chain of variables with supports that depend on their parents.
model = @pgm Chain begin
    let a1 ~ DiscreteUniform(1, 10),
        a2 ~ DiscreteUniform(a1, 10 + a1),
        b1 ~ DiscreteUniform(1, 10),
        b2 ~ DiscreteUniform(b1, 10 + b1),
        x ~ DiscreteUniform(-a2, b2)

        DiscreteUniform(x - 5, x + 5) ↦ 7
        a1
    end
end

# After calibration, query computes the joint posterior of any set of variables by passing messages only within
# the smallest subtree that contains them. We compare it with summing out the full joint distribution (small model)
# and with variable elimination for these variables.
joint, evidence = get_joint_factor(model)
result = junction_tree_message_passing(model, calibrate_tree=true)
println()
for addresses in [[:a1, :b1], [:a2, :x], [:b2, :x]]
    variables = parse_variables(model, addresses)
    q = query(result, variables)
    from_joint = exp.(factor_sum(joint, setdiff(1:model.n_latents, variables)).table) ./ evidence
    ve = greedy_variable_elimination(model, marginal_variables=variables)
    println("query ", addresses, ": equals joint ", exp.(q.factor.table) ./ q.evidence ≈ from_joint,
        ", equals VE ", exp_normalised_table(q.factor) ≈ exp_normalised_table(ve.factor))
end

# Conditioning a factor on observed values selects a slice of its table. This is the same as multiplying with
# indicator factors and summing out the observed variables.
function factor_condition(factor::FactorNode, evidence::Dict{VariableNode,Int})
    variables = [v for v in factor.neighbours if !haskey(evidence, v)]
    table = factor.table[[get(evidence, v, Colon()) for v in factor.neighbours]...]
    return FactorNode(variables, table)
end
function indicator(v::VariableNode, i::Int)
    table = fill(-Inf, length(v.support))
    table[i] = 0.
    return FactorNode([v], table)
end
Random.seed!(0)
all_equal = true
for _ in 1:200
    observed = [v for v in joint.neighbours if rand() < 0.5]
    isempty(observed) && continue
    values = [rand(1:length(v.support)) for v in observed]
    product = reduce(factor_product, indicator.(observed, values), init=joint)
    global all_equal &= factor_sum(product, observed).table ≈ factor_condition(joint, Dict(zip(observed, values))).table
end
println("conditioning = product with indicators and summing out (200 random tests): ", all_equal)


# ===== Sampling from a calibrated junction tree =====
# Exact samples from the posterior: clusters are visited from the root, and each samples its remaining variables
# conditioned on the values sampled before. We compare the sample frequencies of a with the posterior marginal.
model = @pgm Branching begin
    let a ~ DiscreteUniform(1, 3),
        b1 ~ DiscreteUniform(0, 2 * a),
        b2 ~ DiscreteUniform(0, 2 * a),
        c ~ DiscreteUniform(0, (b1 + b2) ÷ 2)

        {:d} ~ DiscreteUniform(c - 1, c + 1) ↦ 5
        a
    end
end

result = junction_tree_message_passing(model, return_factor_as_root=true, calibrate_tree=true)
posterior = get_posterior(result)
a = only(parse_variables(model, [:a]))
Random.seed!(0)
counts = zeros(3)
for _ in 1:10_000
    sample, P = sample_junctiontree(result) # variable => index into support, and the posterior probability of the sample
    counts[sample[a]] += 1
end
println("\nP(a | d = 5) = ", round.(posterior, digits=3), ", frequencies in 10000 samples: ", round.(counts ./ 10_000, digits=3))
