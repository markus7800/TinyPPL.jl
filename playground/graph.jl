using TinyPPL.Distributions
using TinyPPL.Graph
import Random

p = Proposal(:x=>Bernoulli(0.), (:x=>:y=>:z)=>Bernoulli(1.));
haskey(p, :x), p[:x]
haskey(p, :x=>:y), p[:x=>:y]
haskey(p, :x=>:y=>:z), p[:x=>:y=>:z]
get(p, :x=>:y=>:z, Bernoulli(0.5))

d = random_walk_proposal_dist(Categorical([0.1, 0.2, 0.7]), 1, 0.5)
rand(d)

f() = false
b = [true]
model = @ppl Flip begin
    let A = {:a} ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ Main.f()),
        C = {:C} ~ Bernoulli(B == 1 ? 0.9 : 0.7)
        
        (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ $(Main.b[1])
        
        A + C
    end
end

model = @ppl Flip2 begin
    function plus(x, y)
        let a = 1, b = 1
            x + y + a - b
        end
    end
    let A ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ false),
        C ~ Bernoulli(B == 1 ? 0.9 : 0.7),
        D = (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ true
        
        plus(A, C)
    end
end

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
retvals'W

@time traces, retvals = lmh(model, 1_000_000);
mean(retvals)

model.logpdf([1., 0., 1., 1.])

model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = [1., 2., 3., 4., 5.],
        ys = [2.1, 3.9, 5.3, 7.7, 10.2],
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:5]
        
        (slope, intercept)
    end
end
@time traces, retvals = lmh(model, 1_000_000);

kernels = compile_lmh(model, static_observes=true);
kernels = compile_lmh(model, static_observes=true, proposal=Proposal(:intercept=>Normal(0.,1.)));
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);

mean([r[1] for r in retvals])
mean([r[2] for r in retvals])

xs = [
    1., 2., 3, 4., 5., 6., 7., 8., 9., 10.,
    11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
    31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
    41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
    51., 52., 53., 54., 55., 56., 57., 58., 59., 60.,
    61., 62., 63., 64., 65., 66., 67., 68., 69., 70.,
    71., 72., 73., 74., 75., 76., 77., 78., 79., 80.,
    81., 82., 83., 84., 85., 86., 87., 88., 89., 90.,
    91., 92., 93., 94., 95., 96., 97., 98., 99., 100.
];
ys = [
    0.77, 3.94, 5.6, 9.0, 8.95,
    12.17, 11.3, 12.88, 17.56, 18.14,
    20.73, 23.58, 26.16, 27.29, 28.56,
    31.13, 33.17, 33.93, 35.74, 39.2,
    39.93, 41.62, 45.95, 46.71, 49.33,
    50.5, 53.22, 53.34, 58.93, 57.95,
    62.72, 62.33, 65.26, 67.64, 68.72,
    71.02, 71.97, 75.46, 77.28, 79.45,
    80.48, 81.44, 84.04, 88.26, 87.63,
    90.74, 94.25, 94.74, 98.32, 98.9,
    102.86, 103.23, 103.37, 107.81, 109.07,
    111.13, 112.1, 115.76, 117.86, 118.46,
    120.42, 122.09, 123.63, 127.66, 129.04,
    131.88, 134.79, 133.64, 135.52, 139.48,
    141.27, 144.24, 146.11, 148.85, 148.87,
    150.59, 152.62, 154.5, 156.36, 161.28,
    161.85, 162.79, 164.86, 166.88, 169.47,
    170.91, 172.45, 174.53, 177.52, 178.31,
    181.65, 182.27, 184.36, 187.1, 190.0,
    191.64, 193.24, 195.1, 196.71, 199.2
];

model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = $(Main.xs),
        ys = $(Main.ys),
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [{:y=>i} ~ Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i] for i in 1:100]
        
        (slope, intercept)
    end
end;

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000); # 60s
W = exp.(lps);
slope = [r[1] for r in retvals]; slope'W
intercept = [r[2] for r in retvals]; intercept'W

lw = compile_likelihood_weighting(model, static_observes=true)

@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000, static_observes=true); # 1.2s

proposal = Proposal(:slope=>Normal(2.,1.), :intercept=>Normal(-1.,1.));

@time traces, retvals = lmh(model, 1_000_000);
@time traces, retvals = lmh(model, 1_000_000, proposal=proposal); # 60s

kernels = compile_lmh(model, static_observes=true);
kernels = compile_lmh(model, static_observes=true, proposal=proposal);
kernels = compile_lmh(model, [:y], static_observes=true, proposal=proposal);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true); # 2s


mean([r[1] for r in retvals])
mean([r[2] for r in retvals])

@time traces, retvals, lps = hmc(model, 10_000, 0.05, 10, [1. 0.; 0. 1.]);
mean(retvals)

model = @ppl simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end

model = @ppl NormalChain begin
    let x ~ Normal(0., 1.),
        y ~ Normal(x, 1.),
        z = (Normal(y, 1.) ↦ 0)
        x
    end
end

X = Vector{Float64}(undef, model.n_variables);
model.logpdf(X)

import Tracker
X = Tracker.param(rand(3))
lp = model.logpdf(X)
Tracker.back!(lp);
Tracker.grad(X)
X_data = Tracker.data(X)

using TinyPPL.Distributions
function compare_lp(y, x, z)
    return logpdf(Normal(0., 1.), x) + logpdf(Normal(x, 1.), y) + logpdf(Normal(y, 1.), z)
end

X_tracked = Tracker.param.(X_data)
lp_2 = compare_lp(X_tracked...)
Tracker.back!(lp_2)
Tracker.grad.(X_tracked)

@time hmc(model, 1_000, 0.05, 10, [1. 0.; 0. 1.]);

include("../examples/univariate_gmm/data.jl");

t0 = time_ns()
model = @ppl plated GMM begin
    function dirichlet(δ, k)
        let w = [{:w=>i} ~ Gamma(δ, 1) for i in 1:k]
            w / sum(w)
        end
    end
    let λ = 3, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
        δ ~ Uniform(5.0-0.5, 5.0+0.5),
        k = 4,
        y = $(Main.gt_ys[1:3]),
        n = length(y),
        w = dirichlet(δ, k),
        X ~ Categorical(w),
        Y ~ Normal(X, 1.),
        means = [{:μ=>j} ~ Normal(ξ, 1/sqrt(κ)) for j in 1:k],
        vars = [{:σ²=>j} ~ InverseGamma(α, β) for j in 1:k],
        z = [{:z=>i} ~ Categorical(w) for i in 1:n]

        [{:y=>i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]
        
        means
    end
end;
(time_ns() - t0) / 1e9

Random.seed!(0)
X = Vector{Float64}(undef, model.n_variables);
model.sample(X)
model.return_expr(X)
model.logpdf(X) # -3034.9080970776577
for x in X
    println(x)
end

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);

@time lw = compile_likelihood_weighting(model)
@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000; static_observes=true); # 42s
W = exp.(lps);

lps[argmax(lps)]
retvals[argmax(lps)]

addr2var = Addr2Var(:μ=>0.5, :σ²=>2., :w=>5., :z=>1000.)

@time traces, retvals = lmh(model, 1_000_000); # 18s, 90s with full lp computation
@time traces, retvals = rwmh(model, 1_000_000, addr2var=addr2var);

@time kernels = compile_rwmh(model, static_observes=true, addr2var=addr2var);
@time kernels = compile_lmh(model, static_observes=true);
Random.seed!(0);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);
# 0.300861

spgm, E = Graph.to_human_readable(model.symbolic_pgm, model.symbolic_return_expr, model.sym_to_ix);


pgm, plates, plated_edges = Graph.plate_transformation(model, [:w, :μ, :σ², :z, :y]);

@ppl obs begin
    let z ~ Bernoulli(0.5),
        μ0 ~ Normal(-1.0, 1.0),
        μ1 ~ Normal(1.0, 1.0),
        y = 0.5

        if z
            Normal(μ0, 1) ↦ y
        else
            Normal(μ1, 1) ↦ y
        end

    end
end


function test()
    x = 0
    for i in [1,2,3,3,2,1,1,2]
        x += i
    end
    x
end
@code_llvm test()

@time test()



using TinyPPL.Graph
import Random

model = @ppl ObsProg begin
    function or(x, y)
        max(x, y)
    end
    let x ~ Bernoulli(0.6),
        y ~ Bernoulli(0.3)
        Dirac(or(x,y)) ↦ 1
        x
    end
end

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
retvals'W
p = [0.6, 0.12] / 0.72

# P(X = 1 | X || Y = 1)  = P( X = 1,  X || Y = 1) / P(X || Y = 1) = P(X = 1) / (1 - P(X=0, Y=0))
0.6 / (1 - 0.4*0.7)

model = @ppl gf begin
    function or(x, y)
        max(x, y)
    end
    function f(x)
        let flip ~ Bernoulli(0.5),
            y = or(x, flip)

            Dirac(y) ↦ 1
            y
        end
    end
    function g(x)
        1
    end
    let x ~ Bernoulli(0.1),
        obs = f(x)
        x
    end
end
@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
retvals'W


model = @ppl plated Geometric begin
    let N = 25,
        p = 0.5,
        v = [{:x=>i} ~ Bernoulli(p) for i in 1:N],
        g = argmax(v) # = findfirst for floats 0. 1.

        {:X} ~ Normal(g, 1.) ↦ 5.
        g
    end
end


@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
[sum(W[retvals .== i]) for i in 1:10]

lw = compile_likelihood_weighting(model, static_observes=true)
@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000, static_observes=true); # 1.2s


@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);
W = exp.(lps);
retvals'W
# 1/3, 2/3, 9/569

@time f = variable_elimination(model)
evaluate_return_expr_over_factor(model, f)

using TinyPPL.Graph

model = @ppl Survey begin
    let A ~ Categorical([0.3, 0.5, 0.2]), # yound adult old
        S ~ Categorical([0.6, 0.4]), # M, F
        E = if (A == 1 && S == 1) # high, uni
            Categorical([0.75, 0.25])
        elseif (A == 2 && S == 1)
            Categorical([0.72, 0.28])
        elseif (A == 3 && S == 1)
            Categorical([0.88, 0.12])
        elseif (A == 1 && S == 2)
            Categorical([0.64, 0.36])
        elseif (A == 2 && S == 2)
            Categorical([0.7, 0.3])
        elseif (A == 3 && S == 2)
            Categorical([0.9, 0.1])
        end ↦ 1,
        R ~ if (E == 1) # small, big
            Categorical([0.25, 0.75])
        else
            Categorical([0.2, 0.8])
        end,
        O ~ if (E == 1) # emp, self
            Categorical([0.96, 0.04])
        else
            Categorical([0.92, 0.08])
        end,
        T ~ if (O == 1 && R == 1) # car, train, other
            Categorical([0.48, 0.42, 0.1])
        elseif (O == 2 && R == 1)
            Categorical([0.56, 0.36, 0.08])
        elseif (O == 1 && R == 2)
            Categorical([0.58, 0.24, 0.18])
        elseif (O == 2 && R == 2)
            Categorical([0.7, 0.21, 0.09])
        end
        
        (S, T)
    end
end


model = @ppl Student begin
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

include("../examples/exact_inference/burglary.jl")
model = get_model()
is_tree(model)

@time traces, retvals, lps = likelihood_weighting(model, 10_000_000);
W = exp.(lps);

# S = M, T = car | E = high 
sum(W[[r == (1., 1.) for r in retvals]])
0.3427

variable_nodes, factor_nodes = get_factor_graph(model);
for v in variable_nodes
    println(v, ": ", v.support)#, ", ", v.neighbours)
end
for v in factor_nodes
    println(v, ": ")
    @assert all(v.table .<= 0)
end

root = BeliefNode(variable_nodes[5], nothing);
print_belief_tree(root)

return_factor = add_return_factor!(model, variable_nodes, factor_nodes)
root = BeliefNode(return_factor, nothing)
print_belief_tree(root)

f, evidence = belief_propagation(model, false)
f, evidence, marginals = belief_propagation(model, true)

# for (_, address, table) in marginals
#     println(address, ": ", table / sum(table), " ",  sum(table))
# end

@time f = variable_elimination(model)
sum(exp, f.table)
exp.(f.table) / sum(exp, f.table)

evaluate_return_expr_over_factor(model, f)

junction_tree, root_factor = get_junction_tree(model)
f, evidence = junction_tree_message_passing(model)
return_factor, evidence, marginals = junction_tree_message_passing(model, true)
for (_, address, table) in marginals
    println(address, ": ", table)
end

variable_nodes, factor_nodes = get_factor_graph(model)
return_factor = add_return_factor!(model, variable_nodes, factor_nodes)
elimination_order = variable_nodes[[1, 2, 5, 4, 3, 8, 7, 6]]

junction_tree, root_factor = get_junction_tree(factor_nodes, elimination_order, return_factor)

junction_tree_message_passing(model, junction_tree)

for cluster_node in junction_tree
    println(cluster_node, " ", cluster_node.neighbours, " ", cluster_node.factors)
end


using TinyPPL.Graph
N = 1000
# model = @ppl Diamond begin
@time model = Graph.ppl_macro(Set{Symbol}([:uninvoked]), :Diamond, :(begin
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
    @iterate(1000, func, 1.)
end));

# HELLO WORLD, I AM MARKUS
# [8,5,12,12,15, 23,15,18,12,4, 9, 1,13, 13,1,18,11,21,19] 

(exp.(W) / sum(exp, W))[1]
(exp.(res.table) / sum(exp, res.table))[1]
sum(abs, (exp.(W) / sum(exp, W)) .- exp.(res.table) / sum(exp, res.table))

@time variable_nodes, factor_nodes = get_factor_graph(model);
var_to_node = Dict(v.variable => v for v in variable_nodes);

@time res, evidence = belief_propagation(model)

@time res = variable_elimination(model, order=:Topological)
exp.(res.table)
sum(exp, res.table)
exp.(res.table) / sum(exp, res.table)

marginal_variables = return_expr_variables(model)
elimination_order = get_elimination_order(model, variable_nodes, marginal_variables, :WeightedMinFill);
elimination_order = get_elimination_order(model, variable_nodes, marginal_variables, :Topological);
elimination_order = get_elimination_order(model, variable_nodes, marginal_variables, :MinNeighbours);
@time variable_elimination(factor_nodes, elimination_order)

elimination_order = [var_to_node[v] for v in model.topological_order if !(v in marginal_variables)];

return_factor = add_return_factor!(model, variable_nodes, factor_nodes)
return_factor.neighbours
order = [var for var in variable_nodes if !(var in return_factor.neighbours)];
res = variable_elimination(model, factor_nodes, order)

@time traces, retvals, lps = likelihood_weighting(model, 1_000_000);

lw = compile_likelihood_weighting(model, static_observes=true)
@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000, static_observes=true);

retvals'exp.(lps)

p_d = 0.001
p_f = 0.6
function T(P)
    t = (P[2,2] + P[2,1]) * (1 - p_d) + P[1,2]
    return [
        (P[1,1] + (P[2,2] + P[2,1]) * p_d) (1-p_f)*t;
        (p_f * t) 0
    ]
end
T(T(T([0 0; 1 0])))
repeatf(500, T, [0 0; 1 0])

variable_nodes, factor_nodes = get_factor_graph(model, logscale=false);
to_net_file("/Users/markus/Documents/AQUA/diamond_$N.net", variable_nodes, factor_nodes)
to_bif_file("/Users/markus/Documents/AQUA/diamond_$N.bif", variable_nodes, factor_nodes)
return_expr_variables(model)


function to_net_file(path, variable_nodes, factor_nodes)
    open(path, "w") do io
        println(io, "net\n{\n}")
        for v in variable_nodes
            println(io, "node X", v.variable)
            println(io, "{")
            print(io, "  states = ( ")
            for s in v.support
                print(io, "\"", Int(s), "\" ")
            end
            println(io, ");")
            println(io, "}")
        end
        function print_prob_recurse(vars::Vector{VariableNode}, f::FactorNode, index=Int[])
            if isempty(vars)
                @assert !isempty(index)
                print(io, "(")
                print(io, join(f.table[index..., :], " "))
                print(io, ")")
            else
                v = popfirst!(vars)
                print(io, "(")
                for i in 1:length(v.support)
                    push!(index, i)
                    print_prob_recurse(vars, f, index)
                    pop!(index)
                end
                print(io, ")")
                pushfirst!(vars, v)
            end
        end
        for v in factor_nodes
            print(io, "potential ( X", v.neighbours[end].variable)
            if length(v.neighbours) > 1
                print(io, " | ")
                for n in v.neighbours[1:end-1]
                    print(io, "X", n.variable, " ")
                end
                println(io, ")")
            else
                println(io, " )")
            end
            println(io, "{")
            print(io, "  data = ")
            if length(v.neighbours) > 1
                print_prob_recurse(v.neighbours[1:end-1], v)
                println(io, ";")
            else
                print(io, "(")
                print(io, join(v.table, " "))
                println(io, ");")
            end
            println(io, "}")
        end
    end
end

function to_bif_file(path, variable_nodes, factor_nodes)
    open(path, "w") do io
        println(io, "network unknown {\n}")
        for v in variable_nodes
            print(io, "variable X", v.variable)
            println(io, " {")
            print(io, "  type discrete [ $(length(v.support)) ] { ")
            print(io, join(Int.(v.support), ", "))
            println(io, " };")
            println(io, "}")
        end
        function print_prob_recurse(vars::Vector{VariableNode}, f::FactorNode, index=Int[])
            if isempty(vars)
                @assert !isempty(index)
                @assert length(index) == length(f.neighbours)-1
                print(io, "  (")
                values = map(t -> Int(t[2].support[index[t[1]]]), enumerate(f.neighbours[1:end-1]))
                print(io, join(values, ", "))
                print(io, ") ")
                print(io, join(f.table[index..., :], ", "))
                println(io, ";")
            else
                v = popfirst!(vars)
                for i in 1:length(v.support)
                    push!(index, i)
                    print_prob_recurse(vars, f, index)
                    pop!(index)
                end
                pushfirst!(vars, v)
            end
        end
        for v in factor_nodes
            print(io, "probability ( X", v.neighbours[end].variable)
            if length(v.neighbours) > 1
                print(io, " | ")
                print(io, join(map(x -> "X$(x.variable)", v.neighbours[1:end-1]), ", "))
                print(io, " )")
            else
                print(io, " )")
            end
            println(io, " {")
            if length(v.neighbours) > 1
                print_prob_recurse(v.neighbours[1:end-1], v)
            else
                print(io, "  table ")
                print(io, join(v.table, ", "))
                println(io, ";")
            end
            println(io, "}")
        end
    end
end

expr = Graph.rmlines(:(
    let x = y
        x
    end;
    let y = x
        y
    end;
    let x = 1
        x + y
    end;
    let x = 1
        y
        let y = y + x
            x + y
        end
        y
    end
));
Graph.substitute(Dict{Symbol,Any}(:y=>2), expr)