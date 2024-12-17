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




using TinyPPL.Graph


model = @pgm DiceSum begin
    let dice1 ~ Categorical(fill(1/6,6)),
        dice2 ~ Categorical(fill(1/6,6)),
        dice3 ~ Categorical(fill(1/6,6))
        Dirac(dice1 + dice2 + dice3) ↦ 5.
        (dice1,dice2,dice3)
    end
end

model = @pgm DiceSum begin
    let dice1 ~ Categorical(fill(1/6,6)),
        dice2 ~ Categorical(fill(1/6,6)),
        dice3 ~ Categorical(fill(1/6,6))
        Dirac(dice1 + dice2) ↦ 3.
        Dirac(dice2 + dice3) ↦ 3.
        (dice1,)
    end
end
# p(d1=1,d2=2,d3=1) = 0.5
# p(d1=2,d2=1,d3=2) = 0.5

# p(d1=1,d3=1) = 0.5
# p(d1=2,d3=2) = 0.5

# p(d1=2,d2=1) = 0.5
# p(d1=1,d2=2) = 0.5

joint, evidence = get_joint_factor(model)


res, evidence = variable_elimination(model)
post = exp.(res.table) ./ evidence
evaluate_return_expr_over_factor(model, res)

junction_tree, root_cluster_node, root_factor = get_junction_tree(model)
print_junction_tree(root_cluster_node)

res = belief_propagation(model, calibrate_tree=false)
get_posterior_for_root_factor(res)

res = belief_propagation(model, calibrate_tree=true)
get_marginals_from_calibrated_belief_tree(res)


begin
    N = 100000
    res = sample_from_calibrated_belief_tree(tree, N)
    freq = Dict{Vector{Float64},Float64}()
    for i in 1:N
        X =  res[:,i]
        if !haskey(freq,X)
            freq[X] = 0.
        end
        freq[X] += 1.0/N
    end
    freq
end


res, e = get_return_factor_from_calibrated_belief_tree(model, tree)

exp.(res.table) ./ e


joint = [0.0 0.0 0.16666666666666669 0.0 0.0 0.0; 0.0 0.16666666666666669 0.0 0.0 0.0 0.0; 0.16666666666666669 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.16666666666666669 0.0 0.0 0.0 0.0; 0.16666666666666669 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 0.16666666666666669 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
marginal = [0.49999999999999994, 0.3333333333333332, 0.1666666666666666, 0.0, 0.0, 0.0]

joint .* reshape(marginal, :, 1, 1) .* reshape(marginal, 1, :, 1) .* reshape(marginal, 1, 1, :)




model = @pgm Chain begin
    let a1 ~ DiscreteUniform(1,10),
        a2 ~ DiscreteUniform(a1,10+a1),
        b1 ~ DiscreteUniform(1,10),
        b2 ~ DiscreteUniform(b1,10+b1),
        x ~ DiscreteUniform(-a2,b2)
        DiscreteUniform(x-5,x+5) ↦ 7
        a1
    end
end

model.addresses

joint, evidence = get_joint_factor(model)


log_a1b1 = factor_sum(joint, setdiff(1:model.n_latents, parse_variables(model, Any[:a1, :b1])))
a1b1 = exp.(log_a1b1.table) / evidence
heatmap(a1b1)


log_a2b2 = factor_sum(joint, setdiff(1:model.n_latents, parse_variables(model, Any[:a2, :b2])))
a2b2 = exp.(log_a2b2.table) / evidence
heatmap(a2b2)


log_a1x = factor_sum(joint, setdiff(1:model.n_latents, parse_variables(model, Any[:a1, :x])))
a1x = exp.(log_a1x.table) / evidence
heatmap(a1x)


res, evidence = variable_elimination(model)
post = exp.(res.table) ./ evidence
evaluate_return_expr_over_factor(model, res)
bar(post)


res = junction_tree_message_passing(model, return_factor_as_root=true, calibrate_tree=true)
get_posterior_for_root_factor(res)
get_marginals(res)

print_junction_tree(res.root)
q = parse_variables(model, Any[:b2, :x])
q_res, q_e = query(res, q)
q_res = exp.(q_res.table) / q_e
q_res ≈ exp.(factor_sum(joint, setdiff(1:model.n_latents, q)).table) / evidence
heatmap(q_res)

q_joint, q_e = query(res, collect(1:model.n_latents))
joint.table ≈ q_joint.table

res = belief_propagation(model, calibrate_tree=true)
get_posterior_for_root_factor(res)
get_marginals(res)



joint

y1 = 8
y2 = 17

T = joint.table[:,y1,:,y2,:]


v1 = joint.neighbours[2]
e1 = fill(-Inf, size(v1.support))
e1[y1] = 0
f1 = FactorNode([v1], e1)

v2 = joint.neighbours[4]
e2 = fill(-Inf, size(v2.support))
e2[y2] = 0
f2 = FactorNode([v2], e2)


f = factor_product(f1, f2)
f.table # indicator of evidence
f.table[y1,y2]
A = factor_sum(factor_product(joint, f), [v1, v2])
A.table ≈ T

function factor_condition(factor_node::FactorNode, evidence_ixs::Dict{VariableNode,Int})::FactorNode

    vars = VariableNode[]
    table_sel = []
    for v in factor_node.neighbours
        if haskey(evidence_ixs, v)
            push!(table_sel, evidence_ixs[v])
        else
            push!(table_sel, Colon())
            push!(vars,v)
        end
    end

    table = factor_node.table[table_sel...]
    return FactorNode(vars, table)
end

B = factor_condition(joint, Dict(v1 => y1, v2 => y2))

# B = factor_condition(joint, Dict(v1 => y1, v2 => y2, v3 => y3))

sum(exp, joint.table)
sum(exp, A.table)
sum(exp, B.table)

A.table ≈ B.table

function get_one_hot_factor(v, y)
    e = fill(-Inf, size(v.support))
    e[y] = 0
    return FactorNode([v], e)
end
for _ in 1:10000
    vs = VariableNode[]
    ys = Int[]
    for v in joint.neighbours
        if rand() < 0.5
            push!(vs, v)
            push!(ys, rand(1:length(v.support)))
        end
    end
    if isempty(vs)
        continue
    end
    f = reduce(factor_product, map((t -> get_one_hot_factor(t...)), zip(vs,ys)))
    # println(f)

    A = factor_sum(factor_product(joint, f), vs)
    B = factor_condition(joint, Dict(v => y for (v,y) in zip(vs,ys) ))
    @assert(A.table ≈ B.table)
end


# factor_sum(factor_product(indicatior(X), f(X,Y)), X) == factor_condition(f(X,Y), map(indicator(X)))





@time variable_nodes, factor_nodes = read_bif("examples/bif_models/survey.bif");
@time variable_nodes, factor_nodes = read_bif("examples/bif_models/munin.bif");

@time elimination_order = get_greedy_elimination_order(variable_nodes, Int[]);
junction_tree, root_cluster_node, root_factor = get_junction_tree(variable_nodes, elimination_order, factor_nodes[1]);
@time res = junction_tree_message_passing(junction_tree, root_cluster_node, root_factor, true);

print_junction_tree(root_cluster_node)
print_dot_junction_tree(junction_tree)

import Random
begin
    Random.seed!(0)
    t1 = 0.
    t2 = 0.
    for _ in 1:100
        v1 = rand(variable_nodes)
        v2 = rand(variable_nodes)
        if v1 != v2
            println([v1, v2])
            tq1 = @timed greedy_variable_elimination(variable_nodes, factor_nodes, [v1, v2])
            t1 += tq1.time
            q1 = tq1.value
            tq2 = @timed query(res, [v1, v2])
            t2 += tq2.time
            q2 = tq2.value
            @assert isapprox(q1.evidence, q2.evidence, rtol=1e-5)
        end
    end
    println(t1, " vs ", t2)
end


model = @pgm Indep begin
    let x ~ DiscreteUniform(0,5),
        y ~ DiscreteUniform(0,x),
        a ~ DiscreteUniform(0,5),
        b ~ DiscreteUniform(0,a)
        (x,b)
    end
end
res = junction_tree_message_passing(model, return_factor_as_root=true, calibrate_tree=true)
print_dot_junction_tree(res.junction_tree)
q = query(res, parse_variables(model,[:x,:y]))
exp_normalised_table(q.factor)

q = greedy_variable_elimination(model, marginal_variables=parse_variables(model,[:x,:y]))
exp_normalised_table(q.factor)



model = @pgm Diamond begin
    let a ~ DiscreteUniform(1, 3),
        b1 ~ DiscreteUniform(0, 2*a),
        b2 ~ DiscreteUniform(0, 2*a),
        c11 ~ DiscreteUniform(0, 2*b1),
        c12 ~ DiscreteUniform(0, 2*b1),
        c21 ~ DiscreteUniform(0, 2*b2),
        c22 ~ DiscreteUniform(0, 2*b2),
        d1 ~ DiscreteUniform(0, c11 + c12),
        d2 ~ DiscreteUniform(0, c21 + c21),
        e ~ DiscreteUniform(0, d1 + d2)
        {:f} ~ DiscreteUniform(e-1,e+1) ↦ 5
        a
    end
end

model = @pgm Diamond begin
    let a ~ DiscreteUniform(1, 3),
        b1 ~ DiscreteUniform(0, 2*a),
        b2 ~ DiscreteUniform(0, 2*a),
        c ~ DiscreteUniform(0, (b1 + b2) ÷ 2)
        {:d} ~ DiscreteUniform(c-1,c+1) ↦ 5
        a
    end
end

res = greedy_variable_elimination(model, marginal_variables=parse_variables(model, [:c]))
exp_normalised_table(res.factor)

res = junction_tree_message_passing(model, return_factor_as_root=true, calibrate_tree=true)
print_dot_junction_tree(res.junction_tree)
get_posterior_for_root_factor(res)

q = query(res, parse_variables(model, [:c]))
exp_normalised_table(q.factor)


node = res.junction_tree[3]
# Random.seed!(0)
sample_clusternode(res, node)

Random.seed!(0)
X, messages = sample_junctiontree_naive(res)
for node in res.junction_tree
    I = similar(node.potential)
    I.table .= -Inf
    I.table[[X[v.variable] for v in node.potential.neighbours]...] = 0

    belief = exp.(reduce(factor_product, messages[node], init=factor_product(I,node.potential)).table)

    println(sum(belief))
end

