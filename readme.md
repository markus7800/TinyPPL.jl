
# TinyPPL

Educational implementation of a probabilistic programming language in Julia.

Implementation is inspired by [van de Meent et al.: An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756), [Gen.jl](https://github.com/probcomp/Gen.jl) and [Pyro](https://github.com/pyro-ppl/pyro).

There are not too many abstractions and the implementations of inference algorithms are mostly one file large.

Claude Opus 5.5 was used to polish the package (bug fixes, consistency of the interfaces, performance) and to write tests (see [test/](test/)).
Every code change in the package was reviewed, the tests were not.

## Overview

We restrict the distributions to be univariate for the sake of simplicity, but all algorithms can be easily extended to multivariate distributions.


### Evaluation-Based Approach

In the evaluation-based approach, we transform the source code of a Julia function with the `@ppl` macro.

This adds a sampler and observations as arguments.

Samplers decide what happens at sample statements at evaluation time and are used to implement various inference algorithms.

Observations are used to constrain non-latent variables to data, which creates a conditioned probability model.

We use `Gen.jl` sample syntax `X = {addr} ~ Distribution(args...)`.

For example,

```julia
@ppl function simple(mean::Float64)
    X = {:X} ~ Normal(mean, 1.)
    {:Y} ~ Normal(X, 1.)
    return X
end
```

is (essentially) transformed to

```julia
function simple(mean::Float64, sampler::Sampler, observations=Dict())
    X = let distribution = Normal(mean, 1.),
            # check if we have observation
            obs = get(observations, :X, nothing)
        # let sampler decide what to do, the result is assigned to X
        sample(sampler, :X, distribution, obs)
    end
    let distribution = Normal(X, 1.),
        obs = get(observations, :Y, nothing)
        sample(sampler, :Y, distribution, obs)
    end
    return X
end
simple = UniversalModel(simple)
```
This can be checked with the `@macroexpand` function.
(The macro uses fresh names for `distribution`, `obs` and the function, and calls `TinyPPL.Evaluation.sample` and `Base.get` by their full names,
so the model can use any variable names and only `@ppl` has to be imported.)

A model is called with `model(args::Tuple, sampler, observations)`.
Observations can be any dictionary from addresses to values, e.g. `Observations(:Y => 1.)` or `Dict(:Y => 1.)`.


Interaction with the model is possible by implementing custom samplers.

The most basic sampler, simple returns the observations or samples if necessary.
```julia
function sample(sampler::Forward, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        return obs
    end

    return rand(dist)
end
```

There is also the `static` modifier.

```julia
@ppl static function simple(mean::Float64)
    X = {:X} ~ Normal(mean, 1.)
    {:Y} ~ Normal(X, 1.)
    return X
end
```
This allows more efficient implementation of inference algorithms by assuming that we have a fixed finite number of random variables.

In the evaluation-based approach we have no information nor control over the program structure.

We have to run the model in different contexts to implement inference algorithms.

### Graph-Based Approach

For a graph based approach we need to restrict the source language.

For instance, to avoid an unbounded number of nodes we cannot allow dynamic loops, or recursion.  

And we also assume that each distribution produces `Float64`.

We can write let blocks

```julia
let var1 = val1,
    var2 ~ distribution(args...)

    body

end
```
vectors and tuples
```julia
[expr1, expr2, ...]
(expr1, expr2, ...)
```
static generators
```julia
[expr for ix in range]
```
where we have to be able to statically evaluate `range`,
and first-order functions without recursion
```julia
function name(args...)
    body
end
```

Sample statements have syntax
```
var ~ distribution(args...)
```
and observe statements (which return `value` when evaluated)
```
distribution(args...) ↦ value
```
Data can be inlined or interpolated with `$(data)`, which is evaluated where `@pgm` is called (also local variables are visible).
Addresses, loop ranges and observed values may also refer to global variables of the module where `@pgm` is called.

Functions should be defined in the model block.
Only base Julia functions and distributions are available in distribution expressions,
other functions have to be qualified with their module, e.g. `Main.f` for a function `f` defined in Main.


For example,

```julia
model = @pgm simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end
```

returns a probabilistic graphical model with structure
```
Variables:
x1 ~ Normal(0.0, 1.0)
y2 ~ Normal(x1, 1.0)
Observed:
y2 = 1.0
Dependencies:
x1 → y2
Return expression:
x1
```

A linear regression model can be written as
```julia
xs = [1., 2., 3., 4., 5.]
ys = [2.1, 3.9, 5.3, 7.7, 10.2]
model = @pgm LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = $(xs),
        ys = $(ys),
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:5]

        (slope, intercept)
    end
end
```

with structure

```
x1 ~ Normal(0.0, 10.0)
x5 ~ Normal(0.0, 10.0)
y2 ~ Normal(x5 + x1 * 5.0, 1.0)
y3 ~ Normal(x5 + x1 * 2.0, 1.0)
y4 ~ Normal(x5 + x1 * 3.0, 1.0)
y6 ~ Normal(x5 + x1 * 1.0, 1.0)
y7 ~ Normal(x5 + x1 * 4.0, 1.0)
Observed:
y2 = 10.2
y3 = 3.9
y4 = 5.3
y6 = 2.1
y7 = 7.7
Dependencies:
x1 → y2
x1 → y3
x1 → y4
x1 → y6
x1 → y7
x5 → y2
x5 → y3
x5 → y4
x5 → y6
x5 → y7
Return expression:
(x1, x5)
```

In the graph-based approach we have great control over the model and can sample at each node with
`rand(model.distributions[node](X))`, where `X` is a vector where `X[i]` corresponds to the `i`-th variable (i.e. `xi` or `yi`). The result is stored in `X[node]`.

Note that the function `model.distributions[node]` returns a distribution which depends only on the parents of `node`.
We can sample from the complete model by calling `rand(model.distributions[node](X))` in topological order:

```julia
X = Vector{Float64}(undef, pgm.n_latents) # values of latent variables
W = 0. # log likelihood
for node in pgm.topological_order
    d = get_distribution(pgm, node, X)
    if isobserved(pgm, node)
        W += logpdf(d, get_observed_value(pgm, node))
    else
        X[node] = rand(d)
    end
end
r = get_retval(pgm, X)
```
(This is likelihood weighting, see `src/graph/LW.jl`.)

#### Plated Graphical Models

With the `plated` annotation, we tell the compiler that the model has additional structure which can be inferred from the specified addresses.

For example, if we have addresses `:x => i` then all random variables with address prefix `:x` belong to a plate.

This can be used to optimise compilation.

All model functions (distributions, `sample!`, `logpdf`, the plate functions, ...) are generated at runtime as
[RuntimeGeneratedFunctions](https://github.com/SciML/RuntimeGeneratedFunctions.jl), see [notes/graph_codegen.md](notes/graph_codegen.md).
They have no global names; below, the plate functions are shown with names for readability (in the generated code, the function values are interpolated).

The Gaussian Mixture model
```julia
@pgm plated plated_GMM begin
    function dirichlet(δ, k)
        let w = [{:w=>i} ~ Gamma(δ, 1) for i in 1:k]
            w / sum(w)
        end
    end
    let λ = 3, δ = 5.0, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
        k = ({:k} ~ Poisson(λ-1) ↦ 3) + 1,
        y = $(gt_ys),
        n = length(y),
        w = dirichlet(δ, k),
        means = [{:μ=>j} ~ Normal(ξ, 1/sqrt(κ)) for j in 1:k],
        vars = [{:σ²=>j} ~ InverseGamma(α, β) for j in 1:k],
        z = [{:z=>i} ~ Categorical(w) for i in 1:n]

        [{:y=>i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]
        
        means
    end
end
```
has plate structure
```
plate_symbols: [:w, :z, :μ, :σ², :y]
Plate(w,1:4)
Plate(z,5:104)
Plate(μ,105:108)
Plate(σ²,109:112)
Plate(y,114:213)
InterPlateEdge(Plate(z,5:104)->Plate(y,114:213))
PlateToPlateEdge(Plate(σ²,109:112)->Plate(y,114:213))
PlateToPlateEdge(Plate(μ,105:108)->Plate(y,114:213))
PlateToPlateEdge(Plate(w,1:4)->Plate(z,5:104))
```
For instance, from this we know that `:y => i` only depends on `:z => i`.

The compiled logpdf function then looks like
```julia
function (var"##X#797"::INPUT_VECTOR_TYPE, var"##Y#798"::Vector{Float64}) # logpdf
    var"##lp#870" = 0.0
    var"##lp#870" += plated_GMM_lp_plate_w(var"##X#797", var"##Y#798")
    var"##lp#870" += plated_GMM_lp_plate_μ(var"##X#797", var"##Y#798")
    var"##dist#871" = Poisson(2)
    var"##lp#870" += logpdf(var"##dist#871", var"##Y#798"[1])
    var"##lp#870" += plated_GMM_lp_plate_σ²(var"##X#797", var"##Y#798")
    var"##lp#870" += plated_GMM_lp_plate_z(var"##X#797", var"##Y#798")
    var"##lp#870" += plated_GMM_lp_plate_y(var"##X#797", var"##Y#798")
    var"##lp#870"
end
```
where for instance
```julia
function (var"##X#1312"::INPUT_VECTOR_TYPE, var"##Y#1313"::Vector{Float64}) # plated_GMM_lp_plate_y
    var"##lp#1327" = 0.0
    for var"##i#1328" = 1:100
        var"##loop_dist#1329" = Normal(var"##X#1312"[104 + Int(var"##X#1312"[4 + var"##i#1328"])], sqrt(var"##X#1312"[108 + Int(var"##X#1312"[4 + var"##i#1328"])]))
        var"##lp#1327" += logpdf(var"##loop_dist#1329", var"##Y#1313"[(113 + var"##i#1328") - 112])
    end
    var"##lp#1327"
  end
```

Compare this to the usual spaghetti code
```julia
function (var"##X#1089"::INPUT_VECTOR_TYPE, var"##Y#1090"::Vector{Float64}) # logpdf of unplated GMM
    var"##lp#1093" = 0.0
    var"##dist#1094" = InverseGamma(2.0, 10.0)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[112])
    var"##dist#1094" = Normal(0.0, 1 / sqrt(0.01))
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[105])
    var"##dist#1094" = InverseGamma(2.0, 10.0)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[110])
    var"##dist#1094" = Normal(0.0, 1 / sqrt(0.01))
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[107])
    var"##dist#1094" = Normal(0.0, 1 / sqrt(0.01))
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[108])
    var"##dist#1094" = Gamma(5.0, 1)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[1])
    var"##dist#1094" = InverseGamma(2.0, 10.0)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[111])
    var"##dist#1094" = Gamma(5.0, 1)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[4])
    var"##dist#1094" = Poisson(2)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##Y#1090"[1])
    var"##dist#1094" = Gamma(5.0, 1)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[2])
    var"##dist#1094" = Gamma(5.0, 1)
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[3])
    var"##dist#1094" = Categorical(var"##X#1089"[1:4] / sum(var"##X#1089"[1:4]))
    var"##lp#1093" += logpdf(var"##dist#1094", var"##X#1089"[56])
    var"##dist#1094" = Categorical(var"##X#1089"[1:4] / sum(var"##X#1089"[1:4]))
    ...
```

#### Semantics of Graphical Models

By default if statements in the evaluation-based approach are evaluated lazily and in the graph-based approach eagerly.

The annotation `lazy_ifs` makes the graphical model semantics equivalant to the evaluation-based models.

```julia
model = @pgm eager_branching_model begin
    let b ~ Bernoulli(0.5)
        if b == 1.
            let x ~ Normal(-1,1)
                x
            end
        end
    end
end
model.logpdf([0., -10.], Float64[]) # == -42.11208571376462
model.logpdf([1., -10.], Float64[]) # == -42.11208571376462
```

```julia
model = @pgm lazy_ifs lazy_branching_model begin
    let b ~ Bernoulli(0.5)
        if b == 1.
            let x ~ Normal(-1,1)
                x
            end
        end
    end
end
model.addresses
model.logpdf([0., NaN], Float64[])  # == -0.6931471805599453
model.logpdf([1., -10.], Float64[]) # == -42.11208571376462
```

```julia
@ppl function eval_model()
    b ~ Bernoulli(0.5)
    if b == 1
        x ~ Normal(-1,1)
    end
end
logjoint = Evaluation.make_logjoint(eval_model, (), Observations())
logjoint(UniversalTrace(:b => 0., :x => -10.)) # == -0.6931471805599453
logjoint(UniversalTrace(:b => 1., :x => -10.)) # == -42.11208571376462
```

For PGMs this is achieved by using `Flat` distributions 
```julia
function Distributions.logpdf(::Flat, x::Real)
    return 0.
end
function Distributions.rand(::Flat)
    return NaN
end
```

```
Variables:
x1 ~ if true
    Bernoulli(0.5)
else
    Flat()
end
x2 ~ if true && x1 == 1.0
    Normal(-1, 1)
else
    Flat()
end
```


### Inference Algorithms

| Algorithm | Evaluation-Universal | Evaluation-Static | Graph | Reference
|-----------|:----------------:|:-------------:|:-------:|----------|
|Likelihood-Weighting | X | X | X | [van de Meent et al.](https://arxiv.org/abs/1809.10756)|
|Single-Site Metropolis Hastings| X |  | X | [van de Meent et al.](https://arxiv.org/abs/1809.10756)|
|HMC| X (non-parametric) | X | X | [MCMC Handbook](http://www.mcmchandbook.net/HandbookChapter5.pdf), NP-HMC (Mak et al., 2021)|
|ADVI| X | X | X | [ADVI](https://arxiv.org/pdf/1603.00788.pdf)|
|BBVI| X | X | X | [BBVI](https://arxiv.org/pdf/1401.0118.pdf)|
|Variable Elimination|  |  | X | [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)|
|Belief Propagation|  |  | X | [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/), [Bishop](https://link.springer.com/in/book/9780387310732)|
|Junction Tree Message Passing|  |  | X | [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)|
|Involutive MCMC| X |  |  | [iMCMC](https://arxiv.org/pdf/2006.16653.pdf)|
|SMC| | X | X | [van de Meent et al.](https://arxiv.org/abs/1809.10756)|
|Particle Gibbs| | X | X | [PMCMC](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)|
|PGAS| | X | X | [PGAS](https://arxiv.org/abs/1401.0604)|
|AQUA| | | X | [AQUA](https://www.cs.cornell.edu/~saikatd/papers/aqua-atva21.pdf) |

### Results of Inference Algorithms

The results are consistent for each category of inference algorithm and backend
(`UniversalTraces`, `StaticTraces` or `GraphTraces`, indexed by address: `traces[addr]`, `traces[addr, i]`, `retvals(traces)`, `subset(traces, ixs)`):

- Importance sampling / particle methods (`likelihood_weighting`, `smc`, `light_smc`, `conditional_smc`) return `(traces, logw, log_Z)`,
  where `logw` are the normalised log weights (`sum(exp.(logw)) == 1`) and `log_Z` is the estimate of the log evidence log p(Y).
- MCMC methods (`lmh`, `rwmh`, `hmc`, `non_parametric_hmc`, `imcmc`, `particle_gibbs`, `particle_IMH`, compiled single site) return traces with constrained values.
- Variational inference (`advi`, `advi_meanfield`, `advi_fullrank`, `bbvi`) returns a `VIResult`, from which `sample_posterior(result, n)` draws traces.
- Exact inference (`variable_elimination`, `greedy_variable_elimination`, `belief_propagation`, `junction_tree_message_passing`)
  returns a result with `log_evidence` and `get_posterior(result)`, the normalised posterior table over the query variables
  (`get_marginals(result)` for calibrated belief propagation / junction tree results).


## Installation
Requires Julia ≥ 1.10 (tested on 1.13).
Static SMC uses internals of a Libtask fork, so add the fork first:
```console
(@v1.13) pkg> add https://github.com/markus7800/Libtask.jl#subtape
(@v1.13) pkg> add https://github.com/markus7800/TinyPPL.jl
```

```julia
import Pkg
Pkg.add(url="https://github.com/markus7800/Libtask.jl", rev="subtape")
Pkg.add(url="https://github.com/markus7800/TinyPPL.jl")
```

## Usage

See [examples/README.md](examples/README.md) for an overview of the examples. They have their own environment:
```console
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/universal/pedestrian.jl
```