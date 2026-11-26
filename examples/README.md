# TinyPPL Examples

## Setup

The examples have their own environment, which uses the local TinyPPL checkout (`[sources]` in [Project.toml](Project.toml)).
From the repository root:

```console
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/universal/pedestrian.jl
```

Scripts run top to bottom and print their results. When run as a script, plots are saved to `examples/output/`;
in the REPL (e.g. stepping through a script in VS Code) they are displayed instead (see [common/plotting.jl](common/plotting.jl)).

## Contents

### Evaluation-based, universal models ([universal/](universal/))
Models may instantiate an unbounded number of random variables; inference runs the program in different contexts.

| Script | What it shows | Inference |
|---|---|---|
| [pedestrian.jl](universal/pedestrian.jl) | Introduction to universal models: a random walk with an unbounded number of steps, posterior over the number of steps, custom MH proposals (Mak et al.) | likelihood weighting, LMH, RWMH |
| [discontinuous_vi.jl](universal/discontinuous_vi.jl) | Why reparameterisation gradients fail on discontinuous densities (a step function, and branching on a latent variable from Lee et al.), and a custom guide | ADVI, BBVI, guide with `ReinforceELBO` |
| [non_parametric_hmc.jl](universal/non_parametric_hmc.jl) | HMC on the pedestrian model, whose traces have a varying number of steps (Mak et al.), checked against likelihood weighting | nonparametric HMC |
| [geometric.jl](universal/geometric.jl) | Unbounded loop vs. recursion with `@subppl`, checked against the exact posterior | likelihood weighting |
| [dice_paradox.jl](universal/dice_paradox.jl) | Expected rolls until N sixes (in a row) given all rolls are even, conditioning with `Dirac`, checked against the exact answer | likelihood weighting |
| [milch_urn.jl](universal/milch_urn.jl) | Urn with an unknown number of balls (Milch et al.), posterior over the number of balls checked against the exact posterior, noisy or noise-free observations | likelihood weighting |
| [vi.jl](universal/vi.jl) | Variational inference with a mean-field Q over addresses: linear regression, and a model with a random number of variables | ADVI, guide, BBVI |
| [imcmc.jl](universal/imcmc.jl) | Single-site MH and HMC written as involutive MCMC with auxiliary programs and trace transformations, checked against the exact moments | iMCMC |
| [trace_transform.jl](universal/trace_transform.jl) | Trace transformations with automatic Jacobians: Box-Muller, a transformation with auxiliary variables used as importance sampling proposal, stick-breaking Dirichlet | importance sampling |
| [lda.jl](universal/lda.jl) | Latent Dirichlet allocation with Dirichlet distributions as normalised Gammas, label switching between chains | LMH |

### Evaluation-based, static models ([static/](static/))
`@ppl static` models have a fixed, finite set of random variables, which enables more efficient inference.

| Script | What it shows | Inference |
|---|---|---|
| [models.jl](static/models.jl) | Introduction to static models: address indexing, trace matrix, storing a subset of addresses, checked against the exact posterior and evidence of a Gaussian random walk | likelihood weighting |
| [linear_regression.jl](static/linear_regression.jl) | Bayesian linear regression, every sampling algorithm checked against the exact posterior and evidence | likelihood weighting, LMH, RWMH, HMC, SMC, particle IMH |
| [vi.jl](static/vi.jl) | Variational inference on linear regression: mean-field vs. full-rank, ELBO gradient estimators, custom guide | ADVI, BBVI |
| [hmm.jl](static/hmm.jl) | Hidden Markov model (guessing the weather from activities), checked against the exact posterior | likelihood weighting, SMC |
| [lgss.jl](static/lgss.jl) | Linear Gaussian state space model with a custom SMC proposal, checked against the exact posterior and evidence | SMC, particle Gibbs |
| [constrained_support.jl](static/constrained_support.jl) | Variables whose support depends on other variables, unconstrained vs. constrained HMC, checked against the exact distribution | HMC, ADVI, BBVI |

### Graph-based models ([graph/](graph/))
`@pgm` compiles a restricted language to a probabilistic graphical model.

| Script | What it shows | Inference |
|---|---|---|
| [pgm.jl](graph/pgm.jl) | Introduction to `@pgm` models: model structure, functions and data, plates, `@iterate`/`@loop`, eager vs. lazy branching, calling outside functions | likelihood weighting |
| [linear_regression.jl](graph/linear_regression.jl) | Bayesian linear regression, every sampling algorithm checked against the exact posterior and evidence | likelihood weighting, LMH, RWMH, HMC, SMC, particle IMH |
| [vi.jl](graph/vi.jl) | Variational inference on a correlated chain of normals, Rao-Blackwellised BBVI using the graph structure | ADVI, BBVI |
| [constrained_support.jl](graph/constrained_support.jl) | Variables whose support depends on other variables, checked against the exact distribution | HMC, ADVI, BBVI |
| [lgss.jl](graph/lgss.jl) | Linear Gaussian state space model over 400 steps: likelihood weighting collapses, SMC path degeneracy, particle Gibbs with and without ancestral sampling, custom proposals | likelihood weighting, SMC, particle Gibbs |
| [milch_urn.jl](graph/milch_urn.jl) | Urn with an unknown number of balls (Milch et al.) as one model per number of balls, keeping factors small, noisy or noise-free observations | variable elimination |
| [plated_gmm.jl](graph/plated_gmm.jl) | Gaussian mixture model: plates speed up compilation, compiled inference reproduces the generic chains faster, comparison with an evaluation-based model | likelihood weighting, LMH, RWMH (generic and compiled) |
| [aqua.jl](graph/aqua.jl) | Approximate inference by quantising continuous variables (Huang et al.): a conjugate model, return value distributions, benchmarks from the paper | AQUA (VE, BP, junction tree) |
| [hur_slicing.jl](graph/hur_slicing.jl) | Program slicing (Hur et al.): a slice that ignores what the observation depends on gives the wrong posterior | variable elimination |

### Exact inference ([exact_inference/](exact_inference/))

| Script | What it shows | Inference |
|---|---|---|
| [run.jl](exact_inference/run.jl) | Runs a discrete model from [models/](exact_inference/models/): `julia --project=examples examples/exact_inference/run.jl burglary` | VE, BP, junction tree |
| [factor_operations.jl](exact_inference/factor_operations.jl) | Factor products, divisions and benchmarks | VE, BP, junction tree |
| [bif_networks.jl](exact_inference/bif_networks.jl) | Reads Bayesian networks from the [bnlearn repository](https://www.bnlearn.com/bnrepository/) (downloaded on first use) | — |

### Semantics ([semantics/](semantics/))

| Script | What it shows | Inference |
|---|---|---|
| [conditional_observe.jl](semantics/conditional_observe.jl) | An observation inside a branch shifts the posterior towards the other branch (Kiselyov), universal vs. graph | likelihood weighting, LMH |
| [observe_in_functions.jl](semantics/observe_in_functions.jl) | Two functions that always return 1, one with an observation, give different posteriors (Holtzen et al.) | likelihood weighting |
| [varying_dimension.jl](semantics/varying_dimension.jl) | A variable that only exists in one branch: correct with universal MCMC, wrong when the model is declared static | LMH, RWMH, HMC |

### Case studies ([case_studies/](case_studies/))

| Script | What it shows | Inference |
|---|---|---|
| [rjmcmc_gmm/run.jl](case_studies/rjmcmc_gmm/run.jl) | Gaussian mixture with unknown number of components via split/merge moves | reversible jump as iMCMC |
| [golf/golf.jl](case_studies/golf/golf.jl) | Inferring the parameters of a golf physics simulation | IS, MCMC |

Shared data and plotting helpers live in [common/](common/).
