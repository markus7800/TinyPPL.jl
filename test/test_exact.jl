module ExactInferenceTests
# Exact inference on discrete PGMs and AQUA (src/graph/exact).

using Test, Random, Statistics
using TinyPPL.Distributions
using TinyPPL.Graph
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
    @testset "variable_elimination marginal b" begin
        @test ve_marginal(discrete_network, :b, [0., 1.]) ≈ R.DN_POST_B
    end
    @testset "belief_propagation" begin
        res = belief_propagation(discrete_network; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior_for_root_factor(res) ≈ R.DN_POST_Z
        @test res.log_evidence ≈ R.DN_LOG_EVIDENCE
        @test marginal(get_marginals(res), :b, [0., 1.]) ≈ R.DN_POST_B
    end
    @testset "junction_tree_message_passing" begin
        res = junction_tree_message_passing(discrete_network; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior_for_root_factor(res) ≈ R.DN_POST_Z
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
        @test get_posterior_for_root_factor(res) ≈ R.MO_POST_Z
        @test res.log_evidence ≈ R.MO_LOG_EVIDENCE
        @test [p for (_, p) in get_marginals(res)] ≈ [R.MO_POST_Z]
    end
    @testset "junction_tree_message_passing" begin
        res = junction_tree_message_passing(many_observations; calibrate_tree=true, return_factor_as_root=true)
        @test get_posterior_for_root_factor(res) ≈ R.MO_POST_Z
        @test res.log_evidence ≈ R.MO_LOG_EVIDENCE
        @test [p for (_, p) in get_marginals(res)] ≈ [R.MO_POST_Z]
    end
end

@testset "burglary" begin
    ref = [1 - R.BURGLARY_POST_1, R.BURGLARY_POST_1]
    @test ve_return_probs(burglary, variable_elimination(burglary), [0., 1.]) ≈ ref
    @test ve_return_probs(burglary, greedy_variable_elimination(burglary), [0., 1.]) ≈ ref
    res = junction_tree_message_passing(burglary; return_factor_as_root=true)
    @test get_posterior_for_root_factor(res) ≈ ref
end

# ------------------------------------------------------------------------ AQUA

aqua_mean(result, addr) = ((xs, ps) = result[addr]; sum(xs .* ps) * (xs[2] - xs[1]))

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

end # module ExactInferenceTests
