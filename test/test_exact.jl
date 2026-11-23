module ExactInferenceTests
# Exact inference on discrete PGMs and AQUA (src/graph/exact).

using Test, Random, Statistics
using TinyPPL.Distributions
using TinyPPL.Graph
import Distributions: pdf
import TinyPPL.Graph as G
import ..References as R
using ..TestUtils

# ---------------------------------------------------------------------- models
# see references.jl for the mathematical definitions

const discrete_network = @pgm exact_test_discrete_network begin
    let z ~ Categorical([0.5, 0.3, 0.2]),
        b ~ Bernoulli(z == 1. ? 0.1 : (z == 2. ? 0.5 : 0.9))
        {:o1} ~ Bernoulli(b == 1. ? 0.8 : 0.3) ↦ 1.
        {:o2} ~ Bernoulli(z == 3. ? 0.7 : 0.2) ↦ 0.
        z
    end
end

const chain4 = @pgm exact_test_chain4 begin
    let a ~ Bernoulli(0.6),
        b ~ Bernoulli(a == 1. ? 0.7 : 0.2),
        c ~ Bernoulli(b == 1. ? 0.9 : 0.4),
        d ~ Bernoulli(c == 1. ? 0.3 : 0.8)
        {:o} ~ Bernoulli(b == 1. ? 0.0 : 0.5) ↦ 1.
        d
    end
end

# chain x1 -> ... -> x8 -> y, a good elimination order never creates factors with more than 2 variables (B13)
const chain8 = @pgm exact_test_chain8 begin
    let x1 ~ Bernoulli(0.6),
        x2 ~ Bernoulli(x1 == 1. ? 0.7 : 0.2),
        x3 ~ Bernoulli(x2 == 1. ? 0.9 : 0.4),
        x4 ~ Bernoulli(x3 == 1. ? 0.3 : 0.8),
        x5 ~ Bernoulli(x4 == 1. ? 0.6 : 0.1),
        x6 ~ Bernoulli(x5 == 1. ? 0.2 : 0.7),
        x7 ~ Bernoulli(x6 == 1. ? 0.8 : 0.5),
        x8 ~ Bernoulli(x7 == 1. ? 0.4 : 0.9)
        {:y} ~ Bernoulli(x8 == 1. ? 0.9 : 0.2) ↦ 1.
        x1
    end
end

# observation factor with two latent parents rules out b = 1 (C11)
#   P(e = 1 | o = 1) ∝ 0.4 * 0.5 vs P(e = 0 | o = 1) ∝ 0.6 * 0.2 → 0.625
const impossible_branch = @pgm exact_test_impossible_branch begin
    let a ~ Bernoulli(0.6),
        b ~ Bernoulli(a == 1. ? 0.7 : 0.2),
        e ~ Bernoulli(0.4)
        {:o} ~ Bernoulli(b == 1. ? 0.0 : (e == 1. ? 0.5 : 0.2)) ↦ 1.
        a
    end
end

# a is shared by three clusters {a,b}, {a,c}, {a,d} (junction tree is a chain of three clusters)
const star = @pgm exact_test_star begin
    let a ~ Categorical([0.2, 0.5, 0.3]),
        b ~ Bernoulli(a == 1. ? 0.9 : (a == 2. ? 0.4 : 0.1)),
        c ~ Bernoulli(a == 3. ? 0.8 : 0.3),
        d ~ Bernoulli(a == 2. ? 0.0 : 0.6)
        {:ob} ~ Bernoulli(b == 1. ? 0.7 : 0.2) ↦ 1.
        {:oc} ~ Bernoulli(c == 1. ? 0.1 : 0.6) ↦ 0.
        {:od} ~ Bernoulli(d == 1. ? 0.9 : 0.3) ↦ 1.
        a
    end
end

const two_coins = @pgm exact_test_two_coins begin
    let a ~ Bernoulli(0.5),
        b ~ Bernoulli(0.5)
        {:oa} ~ Bernoulli(a == 1. ? 0.9 : 0.1) ↦ 1.
        {:ob} ~ Bernoulli(b == 1. ? 0.1 : 0.9) ↦ 1.
        a
    end
end

const orphan_observation = @pgm exact_test_orphan_observation begin
    let a ~ Bernoulli(0.5)
        {:o} ~ Bernoulli(0.2) ↦ 1.
        a
    end
end

# data must be referenced via Main: @pgm interpolates in the TinyPPL.Graph module (REVIEW.md J4)
const many_observations = @pgm plated exact_test_many_observations begin
    let ys = $(Main.References.MO_YS),
        z ~ Categorical([0.5, 0.5])
        [({:y => i} ~ Bernoulli(z == 1. ? 0.3 : 0.6) ↦ ys[i]) for i in 1:$(Main.References.MO_N)]
        z
    end
end

# from examples/graph_exact_inference.jl
const burglary = @pgm exact_test_burglary begin
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

const normal_normal = @pgm exact_test_normal_normal begin
    let μ ~ Normal(0., 2.),
        ys = [1.0, 2.0, 1.5]
        [({:y => i} ~ Normal(μ, 1.) ↦ ys[i]) for i in 1:3]
        μ
    end
end

const beta_bernoulli = @pgm exact_test_beta_bernoulli begin
    let p ~ Beta(2., 2.),
        ys = [1., 1., 0., 1., 1.]
        [({:y => i} ~ Bernoulli(p) ↦ ys[i]) for i in 1:5]
        p
    end
end

# Beta(0.5, 0.5) has infinite density at the boundary of its support
const arcsine_bernoulli = @pgm exact_test_arcsine_bernoulli begin
    let p ~ Beta(0.5, 0.5)
        {:y} ~ Bernoulli(p) ↦ 1.
        p
    end
end
const ARCSINE_POST_MEAN = 1.5 / 2 # Beta(1.5, 0.5)

# c has support with only one value, x | y ~ N(0.75, 1/2)
const single_point_support = @pgm exact_test_single_point_support begin
    let c ~ Categorical([1.]),
        x ~ Normal(c, 1.)
        {:y} ~ Normal(x, 1.) ↦ 0.5
        x
    end
end

# p has infinite density at 0 and an unbounded parent (support of p is expanded)
const beta_unbounded_parent = @pgm exact_test_beta_unbounded_parent begin
    let s ~ Gamma(2., 1.),
        p ~ Beta(0.5, 1. + s)
        {:y} ~ Bernoulli(p) ↦ 1.
        p
    end
end
# E[p | y=1] = ∫ Gamma(s; 2, 1) P(y=1|s) E[p|y=1,s] ds / ∫ Gamma(s; 2, 1) P(y=1|s) ds
# with P(y=1|s) = 0.5 / (1.5 + s) and p | y=1, s ~ Beta(1.5, 1 + s)
const BETA_UNBOUNDED_POST_MEAN = let ss = range(1e-6, 60, length=600_001), w = pdf.(Gamma(2., 1.), ss) .* 0.5 ./ (1.5 .+ ss)
    sum(w .* 1.5 ./ (2.5 .+ ss)) / sum(w)
end

# posterior of n is concentrated at the lower bound 0 of the support
const poisson_at_bound = @pgm exact_test_poisson_at_bound begin
    let n ~ Poisson(0.5)
        {:y} ~ Normal(n, 1.) ↦ -1.
        n
    end
end
const POISSON_AT_BOUND_POST = let ns = 0:30, w = pdf.(Poisson(0.5), ns) .* pdf.(Normal.(ns, 1.), -1.)
    w ./ sum(w)
end

# --------------------------------------------------------------------- helpers

"P(return value = v) for the return values `values`, from VE result"
function ve_return_probs(pgm, res, values)
    d = Dict(evaluate_return_expr_over_factor(pgm, res.factor))
    return [get(d, v, 0.) for v in values]
end

"P(addr = v) for v in values from marginals [(VariableNode, probs)]"
function marginal(marginals, addr, values)
    for (node, ps) in marginals
        if node.address == addr
            return [ps[findfirst(==(v), node.support)] for v in values]
        end
    end
    error("$addr not found in marginals")
end

function ve_marginal(pgm, addr, values; kw...)
    res = variable_elimination(pgm; marginal_variables=[addr], kw...)
    @assert length(res.factor.neighbours) == 1
    node = res.factor.neighbours[1]
    ps = exp.(res.factor.table) ./ sum(exp, res.factor.table)
    return [ps[findfirst(==(v), node.support)] for v in values]
end

function bp_with_division(pgm)
    variable_nodes, factor_nodes = get_factor_graph(pgm)
    return_factor = add_return_factor!(pgm, variable_nodes, factor_nodes)
    root = get_blief_tree(return_factor)
    return belief_propagation(root, return_factor, true; with_division=true)
end

function jt_with_division(pgm)
    junction_tree, root, root_factor = get_junction_tree(pgm; return_factor_as_root=true)
    return junction_tree_message_passing(junction_tree, root, root_factor, true; with_division=true)
end

# ------------------------------------------------------------ factor operations

@testset "factor operations" begin
    X = VariableNode(1, :X); X.support = [1., 2.]
    Y = VariableNode(2, :Y); Y.support = [1., 2., 3.]
    Z = VariableNode(3, :Z); Z.support = [1., 2., 3., 4.]
    A = FactorNode([X, Y], log.([0.1 0.2 0.3; 0.4 0.5 0.6]))
    B = FactorNode([Y, Z], log.(reshape(collect(1.:12.), 3, 4) ./ 12))
    C = factor_product(A, B)
    @test [v.variable for v in C.neighbours] == [1, 2, 3]
    @test exp.(C.table) ≈ [exp(A.table[x, y]) * exp(B.table[y, z]) for x in 1:2, y in 1:3, z in 1:4]
    S = factor_sum(C, [Y])
    @test exp.(S.table) ≈ dropdims(sum(exp.(C.table), dims=2), dims=2)
    @test factor_division!(C, A, FactorNode(B.neighbours, similar(B.table))).table ≈ B.table

    @testset "factor_sum is numerically stable" begin # C10
        F = FactorNode([X], [-2000., -2001.])
        @test only(factor_sum(F, [X]).table) ≈ -2000 + log(1 + exp(-1))
    end
end

# simulates variable elimination with `order` and returns the largest number of variables of a multiplied factor
function max_factor_scope(factor_nodes, order)
    scopes = [Set(v.variable for v in f.neighbours) for f in factor_nodes]
    max_scope = 0
    for node in order
        involved = [s for s in scopes if node.variable in s]
        psi = reduce(∪, involved; init=Set{Int}())
        max_scope = max(max_scope, length(psi))
        scopes = [[s for s in scopes if !(node.variable in s)]; [setdiff(psi, [node.variable])]]
    end
    return max_scope
end

@testset "elimination order heuristics" begin # B13
    variable_nodes, factor_nodes = G.get_factor_graph(chain8)
    marginal_variables = G.return_expr_variables(chain8)
    ref = variable_elimination(chain8; order=:Topological)
    for order in (:MinNeighbours, :MinFill, :WeightedMinFill)
        @testset "$order" begin
            elimination_order = G.get_elimination_order(chain8, variable_nodes, marginal_variables, order)
            @test length(elimination_order) == 7
            @test !any(v.variable in marginal_variables for v in elimination_order)
            @test max_factor_scope(factor_nodes, elimination_order) == 2 # no fill edges in a chain
            res = variable_elimination(chain8; order=order)
            @test res.log_evidence ≈ ref.log_evidence
            @test exp.(res.factor.table .- res.log_evidence) ≈ exp.(ref.factor.table .- ref.log_evidence)
        end
    end
    @testset "observed child with two parents $order" for order in (:MinNeighbours, :MinFill, :WeightedMinFill)
        res = variable_elimination(impossible_branch; order=order)
        @test res.log_evidence ≈ variable_elimination(impossible_branch; order=:Topological).log_evidence
    end
end

# ------------------------------------------------------------- per model tests

@testset "discrete_network" begin
    @testset "variable_elimination order=$order" for order in (:Topological, :MinNeighbours, :MinFill, :Greedy)
        res = variable_elimination(discrete_network; order=order)
        @test ve_return_probs(discrete_network, res, [1., 2., 3.]) ≈ R.DN_POST_Z
        @test res.log_evidence ≈ R.DN_LOG_EVIDENCE
    end
    @testset "greedy_variable_elimination" begin
        res = greedy_variable_elimination(discrete_network)
        @test ve_return_probs(discrete_network, res, [1., 2., 3.]) ≈ R.DN_POST_Z
        @test res.log_evidence ≈ R.DN_LOG_EVIDENCE
    end
    @testset "get_posterior of variable_elimination" begin # A1
        res = variable_elimination(discrete_network; marginal_variables=[:z])
        node = only(res.factor.neighbours)
        @test get_posterior(res)[[findfirst(==(v), node.support) for v in (1., 2., 3.)]] ≈ R.DN_POST_Z
    end
    @testset "variable_elimination marginal b" begin
        @test ve_marginal(discrete_network, :b, [0., 1.]) ≈ R.DN_POST_B
    end
    @testset "belief_propagation" begin
        res = belief_propagation(discrete_network; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior(res) ≈ R.DN_POST_Z
        @test res.log_evidence ≈ R.DN_LOG_EVIDENCE
        @test marginal(get_marginals(res), :b, [0., 1.]) ≈ R.DN_POST_B
    end
    @testset "junction_tree_message_passing" begin
        res = junction_tree_message_passing(discrete_network; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior(res) ≈ R.DN_POST_Z
        @test res.log_evidence ≈ R.DN_LOG_EVIDENCE
        @test marginal(get_marginals(res), :b, [0., 1.]) ≈ R.DN_POST_B
    end
end

@testset "chain4 with impossible value" begin
    ref(addr) = R.C4_MARGINALS[addr]
    @testset "variable_elimination" begin
        res = variable_elimination(chain4)
        @test ve_return_probs(chain4, res, [0., 1.]) ≈ ref(:d)
        @test res.evidence ≈ R.C4_EVIDENCE
        @test ve_marginal(chain4, :b, [0., 1.]) ≈ ref(:b)
    end
    @testset "belief_propagation" begin
        res = belief_propagation(chain4; calibrate_tree=true, return_factor_as_root=true)
        @test res.evidence ≈ R.C4_EVIDENCE
        for addr in (:a, :b, :c, :d)
            @test marginal(get_marginals(res), addr, [0., 1.]) ≈ ref(addr)
        end
    end
    @testset "belief_propagation with division" begin # C11
        res = bp_with_division(chain4)
        for addr in (:a, :b, :c, :d)
            @test marginal(get_marginals(res), addr, [0., 1.]) ≈ ref(addr)
        end
    end
    @testset "junction_tree_message_passing" begin
        res = junction_tree_message_passing(chain4; calibrate_tree=true, return_factor_as_root=true)
        @test res.evidence ≈ R.C4_EVIDENCE
        for addr in (:a, :b, :c, :d)
            @test marginal(get_marginals(res), addr, [0., 1.]) ≈ ref(addr)
        end
    end
    @testset "junction_tree_message_passing with division" begin # B3
        res = jt_with_division(chain4)
        for addr in (:a, :b, :c, :d)
            @test marginal(get_marginals(res), addr, [0., 1.]) ≈ ref(addr)
        end
    end
end

@testset "belief propagation with division and impossible values" begin # C11
    res = bp_with_division(impossible_branch)
    @test marginal(get_marginals(res), :e, [0., 1.]) ≈ [0.375, 0.625]
    @test marginal(get_marginals(res), :b, [0., 1.]) ≈ [1., 0.]
    res = belief_propagation(impossible_branch; calibrate_tree=true, return_factor_as_root=true)
    @test marginal(get_marginals(res), :e, [0., 1.]) ≈ [0.375, 0.625]
end

@testset "junction tree message passing with division on three clusters and impossible values" begin # B3
    for (pgm, addrs) in ((impossible_branch, (:a, :b, :e)), (star, (:a, :b, :c, :d)))
        res_div = jt_with_division(pgm)
        res = junction_tree_message_passing(pgm; calibrate_tree=true, return_factor_as_root=true)
        @test res_div.log_evidence ≈ res.log_evidence
        for addr in addrs
            support = addr == :a && pgm === star ? [1., 2., 3.] : [0., 1.]
            m_div = marginal(get_marginals(res_div), addr, support)
            @test all(isfinite, m_div)
            @test m_div ≈ marginal(get_marginals(res), addr, support)
        end
    end
    # reference values from enumeration
    res = jt_with_division(impossible_branch)
    @test marginal(get_marginals(res), :e, [0., 1.]) ≈ [0.375, 0.625]
    @test marginal(get_marginals(res), :b, [0., 1.]) ≈ [1., 0.]
end

@testset "disconnected components" begin # C9
    @testset "two_coins variable_elimination" begin
        res = variable_elimination(two_coins)
        @test res.evidence ≈ R.TC_EVIDENCE
        @test ve_return_probs(two_coins, res, [0., 1.]) ≈ [1 - R.TC_POST_A1, R.TC_POST_A1]
    end
    @testset "two_coins greedy_variable_elimination" begin
        res = greedy_variable_elimination(two_coins)
        @test res.evidence ≈ R.TC_EVIDENCE
    end
    @testset "two_coins junction_tree_message_passing" begin
        res = junction_tree_message_passing(two_coins; calibrate_tree=true, return_factor_as_root=true)
        @test res.evidence ≈ R.TC_EVIDENCE
        @test marginal(get_marginals(res), :b, [0., 1.]) ≈ [1 - R.TC_POST_B1, R.TC_POST_B1]
    end
    @testset "orphan_observation variable_elimination" begin
        res = variable_elimination(orphan_observation)
        @test res.evidence ≈ R.OO_EVIDENCE
        @test ve_return_probs(orphan_observation, res, [0., 1.]) ≈ [0.5, 0.5]
    end
    @testset "orphan_observation greedy_variable_elimination" begin
        @test greedy_variable_elimination(orphan_observation).evidence ≈ R.OO_EVIDENCE
    end
    @testset "orphan_observation junction_tree_message_passing" begin
        res = junction_tree_message_passing(orphan_observation; calibrate_tree=true)
        @test res.evidence ≈ R.OO_EVIDENCE
        @test marginal(get_marginals(res), :a, [0., 1.]) ≈ [0.5, 0.5]
    end
end

@testset "many observations (underflow)" begin # C10
    @testset "variable_elimination" begin
        res = variable_elimination(many_observations)
        @test ve_return_probs(many_observations, res, [1., 2.]) ≈ R.MO_POST_Z
        @test res.log_evidence ≈ R.MO_LOG_EVIDENCE
    end
    @testset "belief_propagation" begin
        res = belief_propagation(many_observations; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior(res) ≈ R.MO_POST_Z
        @test res.log_evidence ≈ R.MO_LOG_EVIDENCE
        @test [p for (_, p) in get_marginals(res)] ≈ [R.MO_POST_Z]
    end
    @testset "junction_tree_message_passing" begin
        res = junction_tree_message_passing(many_observations; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior(res) ≈ R.MO_POST_Z
        @test res.log_evidence ≈ R.MO_LOG_EVIDENCE
        @test [p for (_, p) in get_marginals(res)] ≈ [R.MO_POST_Z]
    end
end

@testset "burglary" begin
    ref = [1 - R.BURGLARY_POST_1, R.BURGLARY_POST_1]
    @test ve_return_probs(burglary, variable_elimination(burglary), [0., 1.]) ≈ ref
    @test ve_return_probs(burglary, greedy_variable_elimination(burglary), [0., 1.]) ≈ ref
    res = junction_tree_message_passing(burglary; return_factor_as_root=true)
    @test get_posterior(res) ≈ ref
end

# ------------------------------------------------------------------------ AQUA

aqua_mean(result, addr) = ((xs, ps) = result[addr]; sum(xs .* ps) * G.aqua_cell_width(xs))

@testset "aqua" begin
    @testset "normal_normal method=$method" for method in (:ve, :bp, :jt)
        result = aqua(normal_normal, 100; method=method)
        @test aqua_mean(result, :μ) ≈ R.NN_POST_MEAN atol = 0.01
    end
    @testset "beta_bernoulli method=$method" for method in (:ve, :bp, :jt)
        result = aqua(beta_bernoulli, 100; method=method)
        @test aqua_mean(result, :p) ≈ R.BB_POST_MEAN atol = 0.01
    end
    @testset "infinite density at boundary" begin # B12
        result = aqua(arcsine_bernoulli, 100)
        @test aqua_mean(result, :p) ≈ ARCSINE_POST_MEAN atol = 0.02
    end
    @testset "support with one value" begin # B12
        for method in (:ve, :bp, :jt)
            result = aqua(single_point_support, 100; method=method)
            @test result[:c] == ([1.], [1.])
            @test aqua_mean(result, :x) ≈ 0.75 atol = 0.01
        end
    end
    @testset "infinite density at boundary with unbounded parent" begin # B12
        result = aqua(beta_unbounded_parent, 100)
        xs, ps = result[:p]
        @test all(0 .< xs .< 1) # grid does not contain bounds
        @test all(isfinite, ps)
        @test aqua_mean(result, :p) ≈ BETA_UNBOUNDED_POST_MEAN atol = 0.02
    end
    @testset "discrete support is not expanded beyond bound" begin # B12
        result = aqua(poisson_at_bound, 100)
        xs, ps = result[:n]
        @test xs[1] == 0.
        @test ps ≈ POISSON_AT_BOUND_POST[1:length(xs)] atol = 1e-4
    end
    @testset "aqua_get_joint" begin # B4
        marginals = aqua(normal_normal, 100)
        xs, ps = aqua_get_joint(normal_normal, 100, marginals, Any[:μ])
        # one support vector per joint variable, ps is density table
        @test length(xs) == 1 && size(ps) == (length(xs[1]),)
        Δ = xs[1][2] - xs[1][1]
        @test sum(ps) * Δ ≈ 1
        @test sum(xs[1] .* ps) * Δ ≈ R.NN_POST_MEAN atol = 0.05
    end
end

@testset "bif" begin # B10
    survey = joinpath(@__DIR__, "..", "examples", "bif_models", "survey.bif")
    tables_by_scope(factor_nodes) = Dict([v.variable for v in f.neighbours] => f.table for f in factor_nodes)

    variable_nodes, factor_nodes = read_bif(survey)
    @test [v.address for v in variable_nodes] == ["A", "S", "E", "O", "R", "T"]
    @test length(factor_nodes) == 6
    _, evidence = get_joint_factor(variable_nodes, factor_nodes)
    @test evidence ≈ 1.
    # P(T = train | O = self, R = small) = 0.36, T is the 6th variable, O the 4th, R the 5th
    T_factor = only(f for f in factor_nodes if [v.variable for v in f.neighbours] == [4, 5, 6])
    @test exp(T_factor.table[2, 1, 2]) ≈ 0.36

    mktempdir() do dir
        # Windows line endings
        crlf = joinpath(dir, "survey_crlf.bif")
        write(crlf, replace(read(survey, String), "\n" => "\r\n"))
        _, factor_nodes_crlf = read_bif(crlf)
        @test tables_by_scope(factor_nodes_crlf) == tables_by_scope(factor_nodes)

        # round trip: write_bif writes probabilities (not log probabilities)
        out = joinpath(dir, "survey_out.bif")
        write_bif(out, variable_nodes, factor_nodes)
        variable_nodes_rt, factor_nodes_rt = read_bif(out)
        @test length(variable_nodes_rt) == 6
        rt = tables_by_scope(factor_nodes_rt)
        for (scope, table) in tables_by_scope(factor_nodes)
            @test rt[scope] ≈ table
        end

        # child declared before its parent -> child is not the last neighbour of its factor
        child_first = joinpath(dir, "child_first.bif")
        write(child_first, """
        network unknown {
        }
        variable C {
          type discrete [ 2 ] { no, yes };
        }
        variable P {
          type discrete [ 3 ] { a, b, c };
        }
        probability ( C | P ) {
          (a) 0.9, 0.1;
          (b) 0.5, 0.5;
          (c) 0.2, 0.8;
        }
        probability ( P ) {
          table 0.2, 0.3, 0.5;
        }
        """)
        variable_nodes, factor_nodes = read_bif(child_first)
        out = joinpath(dir, "child_first_out.bif")
        write_bif(out, variable_nodes, factor_nodes)
        @test occursin("probability ( X1 | X2 )", read(out, String))
        _, factor_nodes_rt = read_bif(out)
        rt = tables_by_scope(factor_nodes_rt)
        for (scope, table) in tables_by_scope(factor_nodes)
            @test rt[scope] ≈ table
        end

        # factor that is not a conditional probability table cannot be written
        f = FactorNode(variable_nodes[1:1], log.([0.3, 0.3]))
        @test_throws ErrorException write_bif(joinpath(dir, "bad.bif"), variable_nodes[1:1], [f])

        # unparsable file is not silently accepted
        bad = joinpath(dir, "bad.bif")
        write(bad, "nothing here")
        @test_throws Exception read_bif(bad)
    end
end


# ---------------------------------------------------------------- deep trees (J7)

# binary Markov chain x1 -> ... -> xN with an observation factor on xN, built directly as factor graph
# (a @pgm with 10^5 variables would take long to compile)
function chain_factor_graph(N)
    vs = [VariableNode(i, Symbol(:x, i)) for i in 1:N]
    for v in vs
        v.support = [0., 1.]
    end
    T = [0.7 0.3; 0.2 0.8] # P(x_i | x_{i-1}), rows x_{i-1}
    fs = FactorNode[FactorNode([vs[1]], log.([0.4, 0.6]))]
    for i in 2:N
        push!(fs, FactorNode([vs[i-1], vs[i]], log.(T)))
    end
    push!(fs, FactorNode([vs[N]], log.([0.2, 0.9])))
    for f in fs, v in f.neighbours
        push!(v.neighbours, f)
    end
    # reference by forward recursion (p is the stationary distribution [0.4, 0.6])
    p = [0.4, 0.6]
    for _ in 2:N
        p = T' * p
    end
    log_evidence = log(sum(p .* [0.2, 0.9]))
    posterior_xN = p .* [0.2, 0.9] ./ sum(p .* [0.2, 0.9])
    return vs, fs, log_evidence, posterior_xN
end

@testset "deep trees do not overflow the stack" begin # J7
    N = 100_000 # recursive implementations overflowed
    @testset "is_tree" begin
        vs, fs, _, _ = chain_factor_graph(N)
        @test is_tree(vs, fs)
        # closing the chain to a cycle
        f = FactorNode([vs[1], vs[N]], zeros(2, 2))
        push!(fs, f); push!(vs[1].neighbours, f); push!(vs[N].neighbours, f)
        @test !redirect_stdout(() -> is_tree(vs, fs), devnull)
    end
    @testset "belief_propagation with_division=$with_division" for with_division in (false, true)
        vs, fs, log_evidence, posterior_xN = chain_factor_graph(N)
        res = belief_propagation(get_blief_tree(fs[1]), fs[1], true; with_division=with_division)
        @test res.log_evidence ≈ log_evidence
        marginals = Dict(v.variable => table for (v, table) in get_marginals(res))
        @test length(marginals) == N
        @test marginals[N] ≈ posterior_xN
    end
    @testset "junction_tree_message_passing with_division=$with_division" for with_division in (false, true)
        vs, fs, log_evidence, posterior_xN = chain_factor_graph(N)
        junction_tree, root, root_factor = get_junction_tree(vs, vs, fs[1])
        res = junction_tree_message_passing(junction_tree, root, root_factor, true; with_division=with_division)
        @test res.log_evidence ≈ log_evidence
        marginals = Dict(v.variable => table for (v, table) in get_marginals(res))
        @test marginals[N] ≈ posterior_xN
        X, _ = sample_junctiontree(res)
        @test length(X) == N
        @test isfinite(G._get_logjoint_density(X, res.root, nothing))
    end
end

@testset "junction tree sampling" begin
    # all samplers return the posterior probability of the sample and sample from the posterior
    res = junction_tree_message_passing(discrete_network; calibrate_tree=true)
    Random.seed!(0)
    for _ in 1:10
        X, P = sample_junctiontree(res)
        @test P ≈ get_posterior_density(res, X)
        X, P = sample_junctiontree_optimised_messaging(res)
        @test P ≈ get_posterior_density(res, X)
        for order in (:dfs, :bfs, :random)
            # naive sampler prints and asserts its invariants
            X, P, _ = redirect_stdout(() -> sample_junctiontree_naive(res, order), devnull)
            @test P ≈ get_posterior_density(res, X)
        end
    end
    z_node = only(v for v in keys(G.get_variable_nodes(res.junction_tree)) if v.address == :z)
    samples = [sample_junctiontree(res)[1][z_node.variable] for _ in 1:20_000]
    @test [mean(samples .== i) for i in 1:3] ≈ R.DN_POST_Z atol = 0.015
end

@testset "printing trees" begin
    junction_tree, root, _ = get_junction_tree(chain8)
    out = mktemp() do path, io
        redirect_stdout(() -> print_junction_tree(root), io)
        close(io)
        readlines(path)
    end
    @test length(out) == length(junction_tree)
    @test !startswith(out[1], " ") && all(startswith(line, "  ") for line in out[2:end])
    variable_nodes, factor_nodes = get_factor_graph(chain8)
    out = mktemp() do path, io
        redirect_stdout(() -> print_belief_tree(get_blief_tree(factor_nodes[1])), io)
        close(io)
        readlines(path)
    end
    @test length(out) == length(variable_nodes) + length(factor_nodes)
end

end # module ExactInferenceTests
