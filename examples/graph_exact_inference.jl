using TinyPPL.Distributions
using TinyPPL.Graph

X = VariableNode(1,:X); X.support = [1.,2.];
Y = VariableNode(2,:Y); Y.support = [1.,2.,3.];
Z = VariableNode(3,:Z); Z.support = [1.,2.,3.,4];
A = FactorNode([X, Y], -rand(2,3))
B = FactorNode([Y, Z], -rand(3,4))
C = factor_product(A,B)

B2 = factor_division!(C, A, FactorNode(B.neighbours, similar(B.table)))
B2.table ≈ B.table

B2 = factor_division!(C, A, FactorNode(C.neighbours, similar(C.table)))
B2.table[1,:,:] ≈ B.table
B2.table[2,:,:] ≈ B.table

A2 = factor_division!(C, B, FactorNode(A.neighbours, similar(A.table)))
A2.table ≈ A.table

A2 = factor_division!(C, B, FactorNode(C.neighbours, similar(C.table)))
A2.table[:,:,1] ≈ A.table
A2.table[:,:,2] ≈ A.table
A2.table[:,:,3] ≈ A.table
A2.table[:,:,4] ≈ A.table

C = factor_product(A, A)
A2 = factor_division!(C, A, FactorNode(A.neighbours, similar(A.table)))

X = VariableNode(1,:X); X.support = [1.,2.];
Y = VariableNode(2,:Y); Y.support = [1.,2.];
Z = VariableNode(3,:Z); Z.support = [1.,2.];
A = FactorNode([X, Y], [-Inf 0; -Inf -Inf])
factor_sum(A, [X]).table
B = FactorNode([Y, Z], [-Inf 0; -Inf -Inf])
factor_sum(B, [Z]).table

C = factor_product(A, B)
C.table
A2 = factor_division!(C, B, FactorNode(A.neighbours, similar(A.table)))
A2.table
A2 = factor_division!(C, B, FactorNode(C.neighbours, similar(C.table)))
A2.table

B2 = factor_division!(C, A, FactorNode(C.neighbours, similar(C.table)))
B2.table

model = @pgm Burglary begin
    function or(x, y)
        max(x, y)
    end
    function and(x, y)
        min(x, y)
    end
    let earthquake ~ Bernoulli(0.0001),
        burglary ~ Bernoulli(0.001),
        alarm = or(earthquake, burglary),
        phoneWorking ~ (earthquake == 1 ? Bernoulli(0.7) : Bernoulli(0.99)),
        maryWakes ~ (
            if alarm == 1 
                if earthquake == 1
                    Bernoulli(0.8)
                else
                    Bernoulli(0.6)
                end
            else
                Bernoulli(0.2)
            end
        ),
        called = and(maryWakes, phoneWorking)

        Dirac(called) ↦ 1.
        burglary
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 989190819/992160802, " P(1)=", 2969983/992160802)
end
print_reference_solution()

variable_nodes, factor_nodes = get_factor_graph(model)

res, evidence = variable_elimination(model, variable_nodes, factor_nodes)
evaluate_return_expr_over_factor(model, res)

res, evidence = variable_elimination(variable_nodes, variable_nodes)

res, evidence = junction_tree_message_passing(model)
evaluate_return_expr_over_factor(model, res)


model = @pgm Student begin
    let C ~ Categorical([1.]),
        D ~ Categorical(C==1. ? 1. : 1.),
        I ~ Categorical([1.]),
        G ~ Categorical(D==1. && I==1. ? 1. : 1.),
        L ~ Categorical(G==1. ? 1. : 1.),
        S ~ Categorical(I==1. ? 1. : 1.),
        J ~ Categorical(L==1. && S==1. ? 1. : 1.),
        H ~ Categorical(G==1. && J==1. ? 1. : 1.)

        J
    end
end

variable_nodes, factor_nodes = get_factor_graph(model)
return_factor = add_return_factor!(model, variable_nodes, factor_nodes)

C = variable_nodes[1]
D = variable_nodes[2]
I = variable_nodes[5]
H = variable_nodes[4]
G = variable_nodes[3]
S = variable_nodes[8]
L = variable_nodes[7]
J = variable_nodes[6]
elimination_order = [C, D, I, H, G, S, L, J]
junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, return_factor, true)
print_junction_tree(root_cluster_node)

function inference(show_results=false; algo=:VE, kwargs...)
    model = get_model()

    if algo == :VE
        f = variable_elimination(model; kwargs...)
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

begin
    include("../exact_inference/caesar.jl")
    model = get_model()
    println(model.name)

    @info "Variable Elimination"
    inference(true,algo=:VE)
    print_reference_solution()
    println()

    all_marginals = (model.name == :Survey)
    if is_tree(model)
        @info "Belief Propagation"
        inference(true,algo=:BP,all_marginals=all_marginals)
        print_reference_solution()
        println()
    else
        @info "Cannot apply Belief Propagation"
        println()
    end
    @info "Junction Tree Message Passing"
    inference(true, algo=:JT, all_marginals=all_marginals)
    print_reference_solution()
end



using TinyPPL.Graph
N = 1000
# model = @ppl Diamond begin
@time model = Graph.pgm_macro(Set{Symbol}([:uninvoked]), :Diamond, :(begin
    function or(x, y)
        max(x, y)
    end
    function and(x, y)
        min(x, y)
    end
    function diamond(s1)
        let route ~ Bernoulli(0.5), # Bernoulli(s1 == 1 ? 0.4 : 0.6),
            s2 = route == 1. ? s1 : false,
            s3 = route == 1. ? false : s1,
            drop ~ Bernoulli(0.001)

            or(s2, and(s3, 1-drop))
        end
    end
    function func(old_net)
        let net ~ Dirac(diamond(old_net))
            net
        end
    end
    @iterate($(Main.N), func, 1.)
end));


@time f = variable_elimination(model, order=:Greedy)
@time f = variable_elimination(model, order=:MinFill) # almost all time spent getting elimination order
evaluate_return_expr_over_factor(model, f)

@time [get_junction_tree(model) for _ in 1:10];

variable_nodes, factor_nodes = get_factor_graph(model)
marginal_variables = return_expr_variables(model)

@time f = greedy_variable_elimination(variable_nodes, marginal_variables)

@time begin
    order = get_greedy_elimination_order(variable_nodes, marginal_variables);
    variable_elimination(variable_nodes, order)
end

# DICE eval
modelname = "diamond"
modelname = "ladder"
N = 5000
include("exact_inference/$modelname.jl")
model = get_model();
variable_nodes, factor_nodes, marginal_variables, return_factor = get_model_factor_graph(N);

modelname = "caesar"
include("exact_inference/caesar.jl")
model = get_model();
variable_nodes, factor_nodes = get_factor_graph(model);
return_factor = add_return_factor!(model, variable_nodes, factor_nodes)
marginal_variables = [node.variable for node in return_factor.neighbours]

is_tree(variable_nodes, factor_nodes)

# if modelname == "diamond"
#     elimination_order = variable_nodes[1:end-1]
# elseif modelname == "ladder"
#     elimination_order = variable_nodes[1:end-2]
# elseif modelname == "caesar"
#     elimination_order = variable_nodes[2:end]
# end
@time elimination_order = get_greedy_elimination_order(variable_nodes, marginal_variables);

@time res, evidence = variable_elimination(variable_nodes, elimination_order)
evaluate_return_expr_over_factor(model, res)
print_reference_solution(N)

@time belief_tree = get_blief_tree(return_factor);
@time res, evidence = belief_propagation(belief_tree, return_factor, false);
evaluate_return_expr_over_factor(model, res)
print_reference_solution(N)

belief_tree = get_blief_tree(return_factor);
@time res, evidence, marginals = belief_propagation(belief_tree, return_factor, true);

belief_tree = get_blief_tree(return_factor);
@time res2, evidence2, marginals2 = belief_propagation(belief_tree, return_factor, true; with_division=true);

for ((i, _, m1), (j,_,m2)) in zip(marginals, marginals2)
    @assert i == j
    @assert m1 ≈ m2
end
# elimination_order = variable_nodes
@time elimination_order = get_greedy_elimination_order(variable_nodes, Int[]);
@time junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, return_factor);
@time res, evidence = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, false);
evaluate_return_expr_over_factor(model, res)
print_reference_solution(N)

@time junction_tree, root_cluster_node, root_factor =  get_junction_tree(variable_nodes, elimination_order, return_factor);
@time res, evidence, marginals = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, true);



@time variable_nodes, factor_nodes = read_bif("examples/bif_models/survey.bif");
@time variable_nodes, factor_nodes = read_bif("examples/bif_models/munin.bif");
return_factor = add_return_factor!(factor_nodes, VariableNode[])
is_tree(variable_nodes, factor_nodes)

@time elimination_order = get_greedy_elimination_order(variable_nodes, Int[]);
@time variable_elimination(variable_nodes, elimination_order)

# FIX: is_tree is not indicative of whether we can construct Belief Tree from empty return factor does not make sense
# check directed vs undirected tree
# @time res, evidence = belief_propagation(return_factor, false)
# @time res, evidence, marginals = belief_propagation(return_factor, true)


@time elimination_order = get_greedy_elimination_order(variable_nodes, Int[]);

# Munin: 1.5 seconds
junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, return_factor);
@time res, evidence = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, false);

# Munin 16.5 seconds
junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, return_factor);
@time res, evidence, marginals = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, true);

# Munin 6.5 seconds
junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, return_factor);
@time res2, evidence2, marginals2 = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, true; with_division=true);

for ((i, _, m1), (j,_,m2)) in zip(marginals, marginals2)
    @assert i == j
    @assert m1 ≈ m2
end