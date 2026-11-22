# TinyPPL.jl review — working list

Status legend: `[ ]` open, `[x]` fixed. ✅ = reproduced by running code; otherwise found by reading.
Reference IDs (e.g. `C3`) are used in test comments.

## 0. Tests
- [x] T1 Test suite against known posteriors (`test/`). Run with `Pkg.test(test_args=[...])` or `julia --project=. test/runtests.jl [distributions|universal|static|graph|exact]`.
  Reference posteriors in `test/references.jl` (closed form / enumeration / quadrature). Test comments reference the IDs below.
  Models: normal_normal (continuous), gamma_scale (constrained parent of observations), beta_bernoulli ((0,1) support),
  discrete_network (discrete, tree), mixed (discrete → continuous), chain (address order ≠ topological order),
  branching / geometric (universal: changing / unbounded traces), linear_regression (2-D, plated + unplated),
  plated_gamma_poisson (plate of non-identical constrained latents), lazy_branching (`lazy_ifs`),
  exact: chain4 (impossible value), two_coins / orphan_observation (disconnected), many_observations (underflow), burglary.
- [ ] T2 Not yet covered by tests: C12, C16 (constrained unvisited branch passed), C18, C19, B6, B10, B11 (file untracked), B13, B14, J6, J7, IMCMC, guides, conditional_smc.
- [x] T3 `test/Project.toml` now only lists test deps (Test, Distributions, ForwardDiff, SpecialFunctions + stdlibs), so `Pkg.test()` works. The previous examples environment (Turing, PyPlot, Plots, …) was moved unchanged to `examples/Project.toml` + `examples/Manifest.toml` (run examples with `--project=examples`).

## C. Inference silently returns wrong results
- [x] C1 ✅ `src/evaluation/universal/HMC.jl` — non-parametric HMC kinetic energy `Σp²` missing `/2` (N(0,1) → mean −1.6, var 8.5). Fixed together with C2 (see C2 for the other NP-HMC fixes).
- [x] C2 ✅ `src/evaluation/universal/HMC.jl` — stale addresses deleted from `X_current` (old trace, already stored in `result`) instead of `X_proposed`. NP-HMC was rewritten with plain `Dict`s; further fixes found while fixing:
  - momentum `P` shared its `Dictionaries` index with `X`, so addresses inserted during the gradient pass got an uninitialised momentum and were never added to `K`;
  - instantiated addresses were drawn from the prior at time t (implicit x0 = x_t − t·ε·p0) and `-log q(x0)` was missing from the initial energy; now x0 ~ q = N(0,1), x_t = x0 + t·ε·p0 and both `-log q(x0)` and `p0²/2` enter H(current);
  - addresses unused by the proposed trace are now scored with `-log q` in H(proposed) (extended target π(X_used) ∏ q(X_unused));
  - `x_initial` (constrained trace) is transformed to unconstrained space (was `todo(x_initial)`, part of B5).
  - pedestrian (`examples/universal_non_parametric_hmc.jl`, acceptance 0.0003): not a sampler bug. The default start (a prior sample) usually walks until distance > 10; lowering the distance below 10 adds steps, so HMC is trapped there. The example now starts from the best of 1000 LW samples with L=10, ε=0.05 (acceptance ≈ 0.6, matches LW reference); the docstring notes this, and the test suite has a pedestrian case.
- [x] C3 ✅ `src/evaluation/static/SMC.jl:121` — after last observe `log_w` unchanged but multiplied into `marginal_lik` again (≈Z²); PIMH uses this ratio. Linear-space `marginal_lik` underflows.
  - fixed: weights are accumulated in log space since the last resampling and the evidence is added once per resampling step and once at the end; static `smc`/`conditional_smc` now return the **log** evidence (like graph `smc`), PIMH accepts with the log ratio;
  - latents sampled after the last observe are now scored on return (with a custom proposal their weight Q was ignored, static analogue of C14);
  - resampling weights are computed with `normalise` (no `exp` underflow).
- [x] C4 ✅ `src/evaluation/static/HMC.jl:28` — `transform_to_constrained!` applied even when `unconstrained=false` (Gamma sample 2 → e²). Constrained HMC with default `x_initial = zeros` starts outside support.
  - fixed: without `unconstrained`, retvals are computed with `StaticForward` (no transform); default `x_initial` is a prior sample (in support) for both modes; `x_initial` is copied before transforming (static part of A6).
- [x] C5 ✅ `src/graph/core/core.jl:324-329` — unconstrained logjoint passes unconstrained parent values to children (graph HMC/ADVI/BBVI with `unconstrained=true`).
  - fixed: the constrained value of each transformed latent is stored in a local variable and substituted for `X[node]` in all later distributions (differentiable, no `setindex!`); for plated models the plate functions read/write `X`, so the generated function works on `copy(X)` and writes constrained values into it (also fixes the B7/A6 input mutation);
  - graph `hmc`: `x_initial` is no longer overwritten, default `x_initial` is a prior sample (as for static `hmc`, C4).
- [x] C6 ✅ `src/graph/core/plates.jl:336-346` — spaghetti plate path applies `to_unconstrained` for `:sample`/`:lp` kinds too.
  - fixed: transform only for `:lp_unconstrained`, `:to_constrained`, `:to_unconstrained` (as in the iid and identity-edge paths). Plated models with non-iid constrained plates can be invoked at compile time again (sampling produced negative Gamma values → random DomainError in Poisson).
- [x] C7 ✅ `src/graph/BBVI.jl:5` — iterates nodes in index order, not topological order.
  - fixed: iterate `pgm.topological_order`. In the chain test, `a` has index 1 and its parent `z` index 2, so `a` was sampled with the previous `z` (E[z] ≈ 0.03 instead of 0.44).
- [x] C8 ✅ `src/graph/core/symbolic_pgm.jl:285-296` — with `lazy_ifs`, no edges from `if` condition to branch samples.
  - fixed: parents are the free variables of `phi ? dist : Flat()` (as for observes). The expression is now simplified with `simplify_if`, so samples outside of branches keep their plain distribution instead of `true ? dist : Flat()`.
- [x] C9 ✅ exact: `variable_elimination.jl:59,80`, `greedy_variable_elimination.jl:259,285`, `junction_tree.jl:123,210` — disconnected factor graphs: empty-scope factors dropped / JT only walks root component (two coins: evidence 0.5 instead of 0.25).
  - fixed: VE and greedy VE collect empty-scope factors (initial ones via `get_scalar_factors(factor_nodes)` and each `tau` of a fully eliminated component) and multiply them into the result. If all variables are eliminated, the result factor now holds the log evidence instead of 0.
  - JT: the trees of the connected components are linked with empty sepsets and empty-scope factors are assigned to a component root; also used by AQUA `:jt`. Removed debug `println` in greedy VE.
- [x] C10 ✅ `src/graph/exact/factor_graph.jl:323` (+VE/BP/JT) — `log.(sum(exp, …))` without max shift; evidence returned linear → `-Inf`/`NaN` with many observations.
  - fixed: `factor_sum` uses a max-shifted `logsumexp(table, dims)`; VE/BP/JT results have a new field `log_evidence` (`evidence` is kept, but may underflow); posteriors, marginals, `evaluate_return_expr_over_factor`, `exp_normalised_table` and AQUA normalise in log space.
  - not converted: JT sampling / density helpers (`junction_tree.jl` ≈ l. 520–840) still multiply linear probabilities and divide by `res.evidence`.
- [x] C11 ✅ `src/graph/exact/belief_propagation.jl:482` — `message .-= …` gives `-Inf - -Inf = NaN`; use `factor_div_op`.
  - fixed: `broadcast(factor_div_op, …)` (0/0 = 0). Reproduced with an observation factor with two latent parents that rules out a value (NaN marginals of the other parent); in chain4 the observation factor is a leaf, so the NaN message was never used.
- [ ] C12 `src/graph/SMC.jl:26,41` — `light_smc` returns pre-resampling weights for resampled particles; linear `marginal_lik`.
- [ ] C13 `src/graph/SMC.jl:117-119` — PGAS indexes `X_ref` with observed node (BoundsError); future latents never written into `X`.
- [ ] C14 `src/graph/SMC.jl:73-84` — SMC with custom proposal: latents after last observe don't update `log_w`; `propose_and_logpdf(q, (addr,X))` passes a tuple.
- [ ] C15 ✅ `src/distributions/random_walk.jl:34,53` — truncated split-normal proposal inconsistent with its `logpdf` within 1e-3 of a bound; `var/2` halves variance for bounded proposals; `DiscreteRWProposer` `% 0` for single category.
- [ ] C16 ✅ `src/evaluation/universal/ADVI.jl:31-33,76` — `rand_and_logpdf(::UniversalMeanField)` runs the model via `TraceSampler` on *unconstrained* values → universal `advi_meanfield`/`bbvi` `sample_posterior` crash for constrained parents (Gamma scale → negative σ). Also: unseen addresses filled with constrained prior draws are then transformed as if unconstrained.
- [ ] C17 ✅ `src/evaluation/universal/LMH.jl:88,116` — Gibbs mode applies `log|X|−log|X'|` correction and uses fixed address list (wrong for universal models).
- [ ] C18 `src/evaluation/universal/IMCMC.jl:58-65` — missing addresses silently drawn from prior without proposal term.
- [ ] C19 `src/graph/core/plates.jl:411-439` — plated topological sort ignores edges to/from single plate members.

## B. Crashes / broken features
- [ ] B1 ✅ `src/distributions/transformed.jl:205` — upper-bounded support uses `supp.lb` instead of `supp.ub`.
- [ ] B2 ✅ `src/logjoint/ADVI.jl:122` — ForwardDiff ELBO grad uses default chunk; >12 params → truncated gradient / DimensionMismatch. ReverseDiff version uses internals.
- [ ] B3 ✅ `src/graph/exact/junction_tree.jl:295` — `backward_with_division(neighbour, parent)` should pass `node` → StackOverflow.
- [x] B4 ✅ `src/graph/exact/aqua.jl:283` — destructures `VariableEliminationResult` (also `examples/exact_inference/exact_inference.jl`). Fixed with C10, also the stale VE/BP/JT destructuring in `examples/graph_exact_inference.jl`.
- [ ] B5 ✅ undefined names: `make_unconstrained_logjoint` w/o `pgm` (`graph/ADVI.jl:54`), `marginal_lik_proposed` (`graph/SMC.jl:257`), `proposal` (`compiled/LMH.jl:57`), ~~`todo` (`universal/HMC.jl:137`)~~ (fixed with C2), `SpecialFunctions.digamma` (`logpdf_grad.jl:91`), static `advi(unconstrained=true)` calls universal-only `transform_to_constrained` (`static/advi.jl:61,87`).
- [ ] B6 ✅ `src/graph/core/symbolic_pgm.jl:217` — procedure inlining returns `G_proc` instead of `G`.
- [ ] B7 ✅ graph models only work with Tracker (`INPUT_VECTOR_TYPE`, plate functions typed `Vector{Float64}`); ~~plated `lp_unconstrained` mutates input~~ (fixed with C5).
- [x] B8 ✅ `src/distributions/variational.jl:277` — `rand(::MeanField, n)` returns `Array{Real}` → graph BBVI `sample_posterior` fails. Fixed with C7: returns `Matrix{Float64}`.
- [ ] B9 ✅ `src/distributions/logpdf_grad.jl` — dead code with bugs: Exponential gradient uses rate not scale; `constrain(::Binomial, ::Tuple{Float64})`; undefined `d` in error; discrete methods take `x::Int` (Bernoulli gives `Bool`).
- [ ] B10 `src/graph/exact/bif.jl` — `write_bif` writes log-probs and assumes child is last neighbour; `read_bif` silently parses 0 blocks with CRLF; file never closed.
- [ ] B11 `src/graph/exact/interval_arithmetic.jl` (untracked) — division by interval containing 0, even powers, negative/non-integer exponents, log/sqrt of negatives, top-level `using Plots`.
- [ ] B12 ✅ `src/graph/exact/aqua.jl:36,60` — bounded supports include endpoints (Beta(.5,.5) → Inf → NaN); `support[2]` BoundsError for length-1 support; threshold loop.
- [ ] B13 `src/graph/exact/elimination_order.jl:45-98` — min-fill/min-degree never update neighbour sets.
- [ ] B14 `src/evaluation/static/guide.jl:69,73` — ForwardDiff/ReverseDiff constructors allocate Tracker buffers.
- [ ] B15 ✅ `src/evaluation/universal/LW.jl:77,80` — completion variant reuses one trace Dict for all samples.
- [ ] B16 ✅ `src/distributions/variational.jl:117-122` — `FullRankGaussian` with a negative diagonal entry in `L` throws `DomainError` (log of negative in PDMats); ADVI full-rank optimisation can produce such `L`.
- [ ] B17 ✅ graph `lmh`/`rwmh` fail on `lazy_ifs` models: `Flat` is not a `Distribution` (`StaticProposal(d)` convert error) and has no `random_walk_proposal_dist`.

## A. Inconsistent APIs
- [ ] A1 Return conventions: LW normalised (universal) vs unnormalised (static, graph, completion); SMC evidence log vs linear (static and graph `smc` now log, graph `light_smc` still linear, C12); `lmh` → traces, `imcmc` → (traces, lps), `non_parametric_hmc` → `Vector{Any}` of Dictionaries without retvals; VE → log-space unnormalised `FactorNode`, BP/JT → linear normalised `Array`. Proposal: one `ImportanceResult(traces, logweights, log_evidence)`.
- [ ] A2 ✅ `Observations = Dict{Any,Real}` is concrete+invariant → `Dict(:y => 1.0)` is a MethodError. Accept `AbstractDict`.
- [ ] A3 Overloads that differ only in 5th positional arg type (`likelihood_weighting(..., completion::Function)` / `(..., trace::Addresses)`).
- [ ] A4 Naming/order drift: `addr2proposal` vs `addr2Proposal`; `L` = leapfrog steps / MC samples / Cholesky; `N` vs `L`; positional `K` in some ADVI functions; universal `bbvi` positional `variational_dists`.
- [ ] A5 `Transform` interface declares `in_support`/`support`, implementations use `in_domain`/`domain`; `image(ComposeTransform)` ignores decreasing transforms.
- [ ] A6 ✅ Mutation without `!`: ~~static and graph `hmc` overwrite `x_initial`~~ (fixed with C4, C5), universal `bbvi` mutates dict, ~~plated `lp_unconstrained` mutates `X`~~ (fixed with C5).
- [ ] A7 `Address = Any` makes `GraphTraces[3]` ambiguous (node index vs address).
- [ ] A8 `retvals` is exported from `Evaluation` but not from `Graph` (it is a separate `Graph.retvals`, not an extension of `TinyPPL.retvals`).
- [ ] A9 `make_logjoint` returns a function (universal, graph) or `(function, addresses_to_ix)` (static).

## J. Julia usage / performance
- [ ] J1 Abstract-typed fields (`f::Function`, `PGM` `Vector{Function}`, `StaticTraces.data::Array{Real}`, `TransformedDistribution`, `ContinuousRWProposer`, `MeanField.dists`, `FactorNode.table::Array{Float64}`).
- [ ] J2 ✅ `@pgm` uses runtime `eval` into `TinyPPL.Graph`: global name collisions (redefining a model mutates old PGMs), world-age errors, user globals resolved in package module.
- [ ] J3 No `rng` argument anywhere.
- [ ] J4 `@ppl` hygiene: `let distribution/obs/value` capture user names; `sample`/`get` resolved in caller module. `@pgm` evaluates `$(...)` interpolations in `TinyPPL.Graph`, so data must be referenced as `$(Main.x)` even from inside another module.
- [ ] J5 ✅(sigmoid) Small: eager `get(trace, addr, rand(dist))`; `log(abs(prod(diag(L))))`; `log1p(exp(x))` overflow in `SigmoidTransform`; `x::Float64`-only `logpdf`s; pervasive `::Real` return annotations.
- [ ] J6 Type piracy: `Base.reshape(::Float64, ::Tuple{})`, `Base.randn(::Type{ReverseDiff.TrackedReal})`, `Base.pop!(::Vector{ReductionSize})` popping from front.
- [ ] J7 Recursive BP/JT/`is_cycle` — stack overflow on long chains.

## P. Packaging / hygiene
- [ ] P1 `Revise` is a runtime dependency (unused in `src`). Since the C2 rewrite, `Dictionaries` is unused in `src` as well.
- [ ] P2 `[compat]` only `julia = "1"`; Libtask pinned to personal fork branch `subtape` using internals; Tracker from `master`.
- [ ] P3 Debug `println`s in library code (`greedy_variable_elimination.jl:295`, `belief_propagation.jl:321`, `is_tree`, AQUA).
- [ ] P4 Dead code: ~100 lines commented in `belief_propagation.jl`, commented plotting scripts, exported-but-undefined `factor_division`, `cached_factors`.
- [ ] P5 Typos: `walk_ppl_sytnax`, `get_topolocial_order`, `subtitute_for_syms`.
- [ ] P6 `readme.md` out of date with return types.
