
# TinyPPL

Educational implementation of a probabilistic programming language in Julia.

Implementation is inspired by [van de Meent et al.: An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756), [Gen.jl](https://github.com/probcomp/Gen.jl) and [Pyro](https://github.com/pyro-ppl/pyro).

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
function simple(mean::Float64, sampler::Sampler, observations::Observations)
    function inner()
        X = let distribution = Normal(mean, 1.),
                # check if we have observation
                obs = get(observations, :X, nothing)
                # let sampler decide what to do
                value = sample(sampler, :X, distribution, obs)
                # assign value to X
                value
        end
        let distribution = Normal(X, 1.),
            obs = get(observations, :Y, nothing)
            value = sample(sampler, :Y, distribution, obs)
            value
        end
        return X
    end

    return inner()
end
```
This can be checked with the `@macroexpand` function.


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

In the evaluation-based approach we have no control over the program structure.

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
Data has to be inlined or referenced with `$(Main.data)`, where `data` is a variable in Main which holds data.

All functions have to be defined in the model block.
Only base Julia functions and distributions are available.


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
model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = $(Main.xs),
        ys = $(Main.ys),
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
for node in pgm.topological_order
    d = pgm.distributions[node](X)

    if observed[node]
        value = pgm.observed_values[node](X)
    else
        value = rand(d)
    end
    X[node] = value
    W += logpdf(d, value) # joint probability
end
r = pgm.return_expr(X)
```

### Inference Algorihms

| Algorithm | Eval-Universal | Eval-Static | Graph | Reference
|-----------|:----------------:|:-------------:|:-------:|----------|
|Likelihood-Weighting | X | X | X | [van de Meent et al.](https://arxiv.org/abs/1809.10756)|
|Single-Site Metropolis Hastings| X |  | X | [van de Meent et al.](https://arxiv.org/abs/1809.10756)|
|HMC|  | X | X | [MCMC Handbook](http://www.mcmchandbook.net/HandbookChapter5.pdf)|
|ADVI| X | X | X | [ADVI](https://arxiv.org/pdf/1603.00788.pdf)|
|BBVI| X | X | X | [BBVI](https://arxiv.org/pdf/1401.0118.pdf)|
|Variable Elimination|  |  | X | [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)|
|Belief Propapagation|  |  | X | [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)|
|Junction Tree Message Passing|  |  | X | [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/)|
|Involutive MCMC| X |  |  | [iMCMC](https://arxiv.org/pdf/2006.16653.pdf)|
|SMC| | X | X | [van de Meent et al.](https://arxiv.org/abs/1809.10756)|
|Particle Gibbs| | X | X | [PMCMC](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)|
|PGAS| | X | X | [PGAS](https://arxiv.org/abs/1401.0604)|


## Installation
```console
(@v1.9) pkg> add https://github.com/markus7800/TinyPPL.jl
```

```julia
import Pkg
Pkg.add("https://github.com/markus7800/TinyPPL.jl")
```

## Usage

See [tutorials](examples/tutorials).