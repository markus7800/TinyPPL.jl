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
| [discontinuous_vi.jl](universal/discontinuous_vi.jl) | Why reparameterisation gradients fail on a discontinuous density, and a custom guide | ADVI, BBVI, guide with `ReinforceELBO` |
| [non_parametric_hmc.jl](universal/non_parametric_hmc.jl) | HMC on a model with varying dimension | NP-HMC |
| [geometric.jl](universal/geometric.jl) | Geometric distribution via loop and recursion, compared to exact posterior | likelihood weighting |
| [dice_paradox.jl](universal/dice_paradox.jl) | "Six in a row" dice paradox | likelihood weighting |
| [milch_urn.jl](universal/milch_urn.jl) | Urn with unknown number of balls (Milch et al.) | likelihood weighting |
| [linear_regression.jl](universal/linear_regression.jl) | Bayesian linear regression vs. analytic posterior | IS, MCMC, VI |
| [advi.jl](universal/advi.jl) | ADVI with mean-field and custom guides | ADVI |
| [bbvi.jl](universal/bbvi.jl) | BBVI on a model with a random number of variables | BBVI |
| [hmm_lgss.jl](universal/hmm_lgss.jl) | Hidden Markov model and linear Gaussian state space model | SMC, particle Gibbs |
| [imcmc.jl](universal/imcmc.jl) | LMH and HMC written as involutive MCMC | iMCMC |
| [trace_transform.jl](universal/trace_transform.jl) | Trace transformations (Box-Muller, Dirichlet) | — |
| [lda.jl](universal/lda.jl) | Latent Dirichlet allocation | LMH |

### Evaluation-based, static models ([static/](static/))
`@ppl static` models have a fixed, finite set of random variables, which enables more efficient inference.

| Script | What it shows | Inference |
|---|---|---|
| [models.jl](static/models.jl) | Introduction to static models: address indexing, trace matrix, storing a subset of addresses, checked against the exact posterior and evidence of a Gaussian random walk | likelihood weighting |
| [advi.jl](static/advi.jl) | ADVI on linear regression with mean-field and custom guides | ADVI |
| [constrained_dynamic_support.jl](static/constrained_dynamic_support.jl) | Variables whose support depends on other variables | ADVI, BBVI, HMC |

### Graph-based models ([graph/](graph/))
`@pgm` compiles a restricted language to a probabilistic graphical model.

| Script | What it shows | Inference |
|---|---|---|
| [pgm.jl](graph/pgm.jl) | Introduction to `@pgm` models | likelihood weighting, MCMC |
| [linear_regression.jl](graph/linear_regression.jl) | Bayesian linear regression vs. analytic posterior | IS, MCMC, VI |
| [bbvi.jl](graph/bbvi.jl) | Variational inference on a chain of normals | ADVI, BBVI |
| [conditioning.jl](graph/conditioning.jl) | Observing deterministic expressions (Holtzen, Dice) | likelihood weighting |
| [constrained_support.jl](graph/constrained_support.jl) | Variables whose support depends on other variables | HMC, ADVI |
| [lgss.jl](graph/lgss.jl) | Linear Gaussian state space model vs. exact posterior | SMC, particle Gibbs, PGAS |
| [milch_urn.jl](graph/milch_urn.jl) | Urn model as factor graph (Milch et al.) | exact inference |
| [plated_gmm.jl](graph/plated_gmm.jl) | Gaussian mixture model, plated vs. unplated compilation | compiled LW, LMH, RWMH |
| [aqua.jl](graph/aqua.jl) | Quantised density approximation | AQUA |
| [hur_slicing.jl](graph/hur_slicing.jl) | Program slicing (Hur et al.) | variable elimination |

### Exact inference ([exact_inference/](exact_inference/))

| Script | What it shows | Inference |
|---|---|---|
| [run.jl](exact_inference/run.jl) | Runs a discrete model from [models/](exact_inference/models/): `julia --project=examples examples/exact_inference/run.jl burglary` | VE, BP, junction tree |
| [factor_operations.jl](exact_inference/factor_operations.jl) | Factor products, divisions and benchmarks | VE, BP, junction tree |
| [bif_networks.jl](exact_inference/bif_networks.jl) | Reads Bayesian networks from the [bnlearn repository](https://www.bnlearn.com/bnrepository/) (downloaded on first use) | — |

### Semantics ([semantics/](semantics/))

| Script | What it shows | Inference |
|---|---|---|
| [conditional_observe.jl](semantics/conditional_observe.jl) | Observations inside branches, universal vs. graph | likelihood weighting |
| [discontinuities.jl](semantics/discontinuities.jl) | Densities with discontinuities from branching | — |

### Case studies ([case_studies/](case_studies/))

| Script | What it shows | Inference |
|---|---|---|
| [rjmcmc_gmm/run.jl](case_studies/rjmcmc_gmm/run.jl) | Gaussian mixture with unknown number of components via split/merge moves | reversible jump as iMCMC |
| [golf/golf.jl](case_studies/golf/golf.jl) | Inferring the parameters of a golf physics simulation | IS, MCMC |

Shared data and plotting helpers live in [common/](common/).
