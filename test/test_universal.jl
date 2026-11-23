module UniversalEvaluationTests
# Evaluation-based inference for universal models (src/evaluation/universal).

using Test, Random, Statistics
using TinyPPL.Distributions
using TinyPPL.Evaluation
import TinyPPL.Evaluation as E
import ..References as R
using ..TestUtils

# ---------------------------------------------------------------------- models
# see references.jl for the mathematical definitions

@ppl function normal_normal()
    μ ~ Normal(0., R.NN_PRIOR_SD)
    for i in 1:3
        {:y => i} ~ Normal(μ, 1.)
    end
    return μ
end
const nn_obs = Observations((:y => i) => R.NN_YS[i] for i in 1:3)

@ppl function gamma_scale()
    s ~ Gamma(2., 1.)
    for i in 1:3
        {:y => i} ~ Normal(0., s)
    end
    return s
end
const gs_obs = Observations((:y => i) => R.GS_YS[i] for i in 1:3)

@ppl function beta_bernoulli()
    p ~ Beta(2., 2.)
    for i in 1:5
        {:y => i} ~ Bernoulli(p)
    end
    return p
end
const bb_obs = Observations((:y => i) => R.BB_YS[i] for i in 1:5)

@ppl function discrete_network()
    z ~ Categorical([0.5, 0.3, 0.2])
    b ~ Bernoulli(R.dn_pb(z))
    {:o1} ~ Bernoulli(R.dn_po1(b))
    {:o2} ~ Bernoulli(R.dn_po2(z))
    return z
end
const dn_obs = Observations(:o1 => 1., :o2 => 0.)

@ppl function mixed()
    z ~ Bernoulli(0.3)
    x ~ Normal(R.mx_mean(z), 1.)
    {:y} ~ Normal(x, 0.5)
    return x
end
const mx_obs = Observations(:y => R.MX_Y)

@ppl function branching()
    b ~ Normal(0., 1.)
    if b > 0
        x = {:c} ~ Normal(1., 1.)
    else
        x = {:d} ~ Normal(-1., 1.)
    end
    {:y} ~ Normal(x, 1.)
    return b > 0
end
const br_obs = Observations(:y => R.BR_Y)

# variant with a constrained latent in one branch (only used to check support handling)
@ppl function branching_constrained()
    b ~ Normal(0., 1.)
    if b > 0
        x = {:c} ~ Gamma(2., 1.)
    else
        x = {:d} ~ Normal(-1., 1.)
    end
    {:y} ~ Normal(x, 1.)
    return b > 0
end

@ppl function geometric()
    i = 0
    while ({:b => i} ~ Bernoulli(0.5)) == 0
        i += 1
    end
    {:y} ~ Normal(i, 1.)
    return i
end
const geo_obs = Observations(:y => R.GEO_Y)

@ppl function linear_regression(xs)
    slope ~ Normal(0., 10.)
    intercept ~ Normal(0., 10.)
    for i in eachindex(xs)
        {:y => i} ~ Normal(intercept + slope * xs[i], 1.)
    end
    return (slope, intercept)
end
const lr_obs = Observations((:y => i) => R.LR_YS[i] for i in eachindex(R.LR_YS))

# -------------------------------------------------------------- deterministic

@testset "logjoint" begin
    lj = E.make_logjoint(gamma_scale, (), gs_obs)
    s = 1.7
    ref = logpdf(Gamma(2., 1.), s) + sum(logpdf(Normal(0., s), y) for y in R.GS_YS)
    @test lj(UniversalTrace(:s => s)) ≈ ref

    ulj = E.make_unconstrained_logjoint(gamma_scale, (), gs_obs)
    @test ulj(UniversalTrace(:s => log(s))) ≈ ref + log(s) # + log |d exp(x)/dx|

    X, _ = transform_to_unconstrained(UniversalTrace(:s => s), gamma_scale, (), gs_obs)
    @test X[:s] ≈ log(s)
    Y, retval = transform_to_constrained(X, gamma_scale, (), gs_obs)
    @test Y[:s] ≈ s
    @test retval ≈ s
end

# @ppl hygiene (J4): only the macro is imported, the module defines its own `sample`, `get` and `param`,
# and the model uses the names of variables that the macro introduces in addresses.
module PPLHygiene
    import TinyPPL.Evaluation: @ppl, @subppl
    import TinyPPL.Distributions: Normal
    sample(xs) = error("user sample")
    get(xs) = error("user get")
    param(xs) = error("user param")

    @ppl function sub(a, b)
        {:sub} ~ Normal(a + b, 1.)
    end
    @ppl function model()
        distribution, obs, value, addr = 1, 2, 3, 4
        x1 = {:x => distribution} ~ Normal(0., 1.)
        x2 = {:x => obs} ~ Normal(0., 1.)
        x3 = {:x => value} ~ Normal(0., 1.)
        x4 = {:x => addr} ~ Normal(0., 1.)
        s = @subppl sub(x1, x2)
        return (x1, x2, x3, x4, s)
    end
    @ppl static function guide()
        m = param("m")
        x ~ Normal(m[1], 1.)
    end
end

@testset "@ppl hygiene" begin # J4
    Random.seed!(0)
    X = sample_trace(PPLHygiene.model, ())
    @test Set(keys(X)) == Set(Any[:sub, :x => 1, :x => 2, :x => 3, :x => 4])
    # observations are looked up with the user's address
    obs = Observations((:x => 2) => 0.5, (:x => 3) => -0.5)
    X, retval = E.sample_trace(PPLHygiene.model, (), obs), PPLHygiene.model((), Forward(), obs)
    @test !haskey(X, :x => 2) && !haskey(X, :x => 3)
    @test retval[2:3] == (0.5, -0.5)
    q = E.make_guide(PPLHygiene.guide, (), Dict{Any,Int}(:x => 1))
    @test length(E.rand(q)) == 1
end

@testset "single-site samplers reuse current value without sampling" begin # J5
    for sampler in (E.LMH(Addr2Proposal()), E.RWMH(1., Addr2Var()))
        sampler.trace_current[:a] = 0.3
        sampler.resample_addr = :b
        Random.seed!(1)
        @test E.sample(sampler, :a, Normal(0., 1.), nothing) == 0.3
        r = rand()
        Random.seed!(1)
        @test r == rand() # no random number was drawn
    end
end

@testset "observations as plain Dict" begin # A2
    Random.seed!(0)
    @test likelihood_weighting(normal_normal, (), Dict((:y => i) => R.NN_YS[i] for i in 1:3), 10) isa Tuple
end

# ---------------------------------------------------------- likelihood weighting

@testset "likelihood_weighting" begin
    N = 50_000
    @testset "normal_normal" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(normal_normal, (), nn_obs, N)
        @test wmean(traces[:μ], lw) ≈ R.NN_POST_MEAN atol = 0.03
        @test wstd(traces[:μ], lw) ≈ R.NN_POST_SD atol = 0.03
        @test wmean(retvals(traces), lw) ≈ R.NN_POST_MEAN atol = 0.03
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(gamma_scale, (), gs_obs, N)
        @test wmean(traces[:s], lw) ≈ R.GS_POST_MEAN atol = 0.05
    end
    @testset "beta_bernoulli" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(beta_bernoulli, (), bb_obs, N)
        @test wmean(traces[:p], lw) ≈ R.BB_POST_MEAN atol = 0.01
    end
    @testset "discrete_network" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(discrete_network, (), dn_obs, N)
        @test [wprob(traces[:z], lw, k) for k in 1:3] ≈ R.DN_POST_Z atol = 0.01
        @test wprob(traces[:b], lw, 1) ≈ R.DN_POST_B[2] atol = 0.01
    end
    @testset "mixed" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(mixed, (), mx_obs, N)
        @test wprob(traces[:z], lw, 1) ≈ R.MX_POST_Z1 atol = 0.01
        @test wmean(traces[:x], lw) ≈ R.MX_POST_MEAN_X atol = 0.03
    end
    @testset "branching" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(branching, (), br_obs, N)
        @test wmean(retvals(traces), lw) ≈ R.BR_POST_POSITIVE atol = 0.01
    end
    @testset "geometric" begin
        Random.seed!(0)
        traces, lw = likelihood_weighting(geometric, (), geo_obs, N)
        @test wmean(retvals(traces), lw) ≈ R.GEO_POST_MEAN atol = 0.05
    end
    @testset "retval completion" begin
        Random.seed!(0)
        rets, lw = likelihood_weighting(normal_normal, (), nn_obs, N, E.retval_completion)
        @test wmean(rets, lw) ≈ R.NN_POST_MEAN atol = 0.03
    end
    @testset "trace completion returns distinct traces" begin # B15
        Random.seed!(0)
        ts, _ = likelihood_weighting(normal_normal, (), nn_obs, 3, (trace, retval) -> trace)
        @test ts[1][:μ] != ts[2][:μ]
        # each kept trace belongs to its own sample (normal_normal returns μ)
        res, _ = likelihood_weighting(normal_normal, (), nn_obs, 5, (trace, retval) -> (trace, retval))
        @test all(trace[:μ] == retval for (trace, retval) in res)
        @test length(unique(objectid(trace) for (trace, _) in res)) == 5
    end
end

# ------------------------------------------------------------------ single-site MH

const MH_MODELS = [
    # name, model, args, observations, test function(traces)
    ("normal_normal", normal_normal, (), nn_obs, t -> begin
        μ = burn(t[:μ])
        @test mean(μ) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(μ) ≈ R.NN_POST_SD atol = 0.05
    end),
    ("gamma_scale", gamma_scale, (), gs_obs, t -> begin
        @test mean(burn(t[:s])) ≈ R.GS_POST_MEAN atol = 0.1
    end),
    ("beta_bernoulli", beta_bernoulli, (), bb_obs, t -> begin
        @test mean(burn(t[:p])) ≈ R.BB_POST_MEAN atol = 0.02
    end),
    ("discrete_network", discrete_network, (), dn_obs, t -> begin
        @test frequencies(burn(t[:z]), 1:3) ≈ R.DN_POST_Z atol = 0.02
        @test mean(burn(t[:b])) ≈ R.DN_POST_B[2] atol = 0.02
    end),
    ("mixed", mixed, (), mx_obs, t -> begin
        @test mean(burn(t[:z])) ≈ R.MX_POST_Z1 atol = 0.03
        @test mean(burn(t[:x])) ≈ R.MX_POST_MEAN_X atol = 0.05
    end),
    ("branching", branching, (), br_obs, t -> begin
        @test mean(burn(retvals(t))) ≈ R.BR_POST_POSITIVE atol = 0.03
    end),
    ("geometric", geometric, (), geo_obs, t -> begin
        @test mean(burn(retvals(t))) ≈ R.GEO_POST_MEAN atol = 0.1
    end),
]

@testset "lmh" begin
    for (name, model, args, obs, check) in MH_MODELS
        @testset "$name" begin
            Random.seed!(0)
            check(lmh(model, args, obs, 50_000))
        end
    end
    @testset "n_updates geometric" begin # C17
        Random.seed!(0)
        traces = lmh(geometric, (), geo_obs, 20_000; n_updates=3)
        @test mean(burn(retvals(traces))) ≈ R.GEO_POST_MEAN atol = 0.1
    end
    @testset "n_updates linear_regression" begin # C17
        Random.seed!(0)
        traces = lmh(linear_regression, (R.LR_XS,), lr_obs, 20_000; n_updates=2, addr2proposal=Addr2Proposal(:slope => ContinuousRandomWalkProposal(0.1), :intercept => ContinuousRandomWalkProposal(0.5)))
        @test [mean(burn(traces[:slope])), mean(burn(traces[:intercept]))] ≈ R.LR_POST_MEAN atol = 0.1
    end
    @testset "random walk proposal normal_normal" begin
        Random.seed!(0)
        traces = lmh(normal_normal, (), nn_obs, 50_000; addr2proposal=Addr2Proposal(:μ => ContinuousRandomWalkProposal(0.5)))
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
    end
end

@testset "rwmh" begin
    for (name, model, args, obs, check) in MH_MODELS
        @testset "$name" begin
            Random.seed!(0)
            check(rwmh(model, args, obs, 50_000; default_var=0.5))
        end
    end
end

# ---------------------------------------------------------------------- iMCMC

# random walk proposal as involution: swap current value of μ and auxiliary value
@ppl function rw_aux(tr::UniversalTrace)
    {:proposed} ~ Normal(tr[:μ], 0.5)
end
function swap_involution!(tt::TraceTransformation, old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)
    write_continuous(tt, new_model_trace, :μ, read_continuous(tt, old_proposal_trace, :proposed))
    write_continuous(tt, new_proposal_trace, :proposed, read_continuous(tt, old_model_trace, :μ))
end
# forgets to write μ
function incomplete_involution!(tt::TraceTransformation, old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)
    write_continuous(tt, new_proposal_trace, :proposed, read_continuous(tt, old_model_trace, :μ))
    write_continuous(tt, new_model_trace, :ν, read_continuous(tt, old_proposal_trace, :proposed))
end
# writes superfluous address
function superfluous_involution!(tt::TraceTransformation, old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)
    swap_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
    write_continuous(tt, new_model_trace, :ν, 0.)
end

@testset "imcmc" begin
    @testset "random walk normal_normal" begin
        Random.seed!(0)
        traces, lps = imcmc(normal_normal, (), nn_obs, rw_aux, (), swap_involution!, 50_000)
        @test mean(burn(traces[:μ])) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(burn(traces[:μ])) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "incomplete trace from involution" begin # C18
        Random.seed!(0)
        @test_throws ErrorException imcmc(normal_normal, (), nn_obs, rw_aux, (), incomplete_involution!, 10)
        @test_throws ErrorException imcmc(normal_normal, (), nn_obs, rw_aux, (), superfluous_involution!, 10)
    end
end

# ---------------------------------------------------------- non-parametric HMC

@testset "non_parametric_hmc" begin
    @testset "normal_normal" begin # C1
        Random.seed!(0)
        result = non_parametric_hmc(normal_normal, (), nn_obs, 5_000, 10, 0.1)
        μ = burn([X[:μ] for X in result])
        @test mean(μ) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(μ) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "branching" begin # C2
        Random.seed!(0)
        result = non_parametric_hmc(branching, (), br_obs, 5_000, 10, 0.1)
        # every trace contains exactly the branch address that matches the sign of b
        @test all(haskey(X, X[:b] > 0 ? :c : :d) && !haskey(X, X[:b] > 0 ? :d : :c) for X in result)
        @test mean(burn([X[:b] > 0 for X in result])) ≈ R.BR_POST_POSITIVE atol = 0.05
        # E[c | b > 0, y] = (1 + y) / 2
        @test mean(X[:c] for X in burn(result) if haskey(X, :c)) ≈ (1 + R.BR_Y) / 2 atol = 0.05
    end
    @testset "gamma_scale" begin # values are unconstrained: s = exp(X[:s])
        Random.seed!(0)
        result = non_parametric_hmc(gamma_scale, (), gs_obs, 5_000, 10, 0.2)
        s = burn([exp(X[:s]) for X in result])
        @test mean(s) ≈ R.GS_POST_MEAN atol = 0.1
        @test std(s) ≈ R.GS_POST_SD atol = 0.1
    end
    @testset "x_initial" begin # B5
        Random.seed!(0)
        result = non_parametric_hmc(normal_normal, (), nn_obs, 10, 10, 0.1; x_initial=UniversalTrace(:μ => 1.0))
        @test length(result) == 10
        result = non_parametric_hmc(gamma_scale, (), gs_obs, 1, 10, 0.2; x_initial=UniversalTrace(:s => 1.0))
        @test haskey(result[1], :s) # x_initial is constrained and must be transformed
    end
end

# ---------------------------------------------------------------- variational

@testset "advi_meanfield" begin
    @testset "normal_normal" begin
        Random.seed!(0)
        # decayed adagrad keeps a constant step size, so a small learning rate is needed for accuracy
        res = advi_meanfield(normal_normal, (), nn_obs, 10_000, 10, 0.01)
        traces, _ = sample_posterior(res, 10_000)
        @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.05
        @test std(traces[:μ]) ≈ R.NN_POST_SD atol = 0.05
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        res = advi_meanfield(gamma_scale, (), gs_obs, 3_000, 10, 0.05)
        traces, _ = sample_posterior(res, 10_000)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2 # Gaussian approximation in log space
    end
    @testset "linear_regression" begin
        Random.seed!(0)
        res = advi_meanfield(linear_regression, (R.LR_XS,), lr_obs, 5_000, 10, 0.05)
        traces, _ = sample_posterior(res, 10_000)
        @test [mean(traces[:slope]), mean(traces[:intercept])] ≈ R.LR_POST_MEAN atol = 0.1
    end
    @testset "addresses without variational distribution are sampled from prior" begin # C16
        Random.seed!(0)
        umf = E.UniversalMeanField(Dict{Any,E.VariationalDistribution}(), gamma_scale, (), gs_obs)
        res = E.UniversalVIResult(umf, X -> transform_to_constrained(X, gamma_scale, (), gs_obs))
        traces, lps = sample_posterior(res, 10_000)
        @test mean(traces[:s]) ≈ 2. atol = 0.05 # prior Gamma(2, 1)
        # lp is the log density of the unconstrained prior sample
        X, lp = E.rand_and_logpdf(umf)
        @test lp ≈ logpdf(Gamma(2., 1.), exp(X[:s])) + X[:s]
    end
    @testset "branching: unvisited addresses" begin # C16
        Random.seed!(0)
        res = advi_meanfield(branching_constrained, (), br_obs, 500, 10, 0.05)
        traces, _ = sample_posterior(res, 1_000)
        # returned traces must be consistent with the program: branch address matching b, in its support
        @test all(X[:b] > 0 ? X[:c] > 0 : haskey(X, :d) for X in traces.data)
    end
end

@testset "bbvi" begin
    @testset "normal_normal" begin
        Random.seed!(0)
        res = bbvi(normal_normal, (), nn_obs, 3_000, 10, 0.05)
        traces, _ = sample_posterior(res, 10_000)
        @test mean(traces[:μ]) ≈ R.NN_POST_MEAN atol = 0.1
        @test std(traces[:μ]) ≈ R.NN_POST_SD atol = 0.1
    end
    @testset "gamma_scale" begin
        Random.seed!(0)
        res = bbvi(gamma_scale, (), gs_obs, 3_000, 10, 0.05)
        traces, _ = sample_posterior(res, 10_000)
        @test mean(traces[:s]) ≈ R.GS_POST_MEAN atol = 0.2
    end
end

end # module UniversalEvaluationTests
