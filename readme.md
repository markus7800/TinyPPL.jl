
# TinyPPL

Minimal implementation of common probabilistic programming language design paradigms in JULIA.

Implementation is inspired by [van de Meent et al.: An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756), [Gen.jl](https://github.com/probcomp/Gen.jl) and [Pyro](https://github.com/pyro-ppl/pyro).

## Overview

We restrict the distributions to be univariate for the sake of simplicity, but all algorithms can be easily extended to multivariate distributions.

Currently implemented:

- [trace-based approach](src/trace/readme.md)  
  - Likelihood weighting
- [evaluation-based approach](src/evaluation/readme.md)  
  - Likelihood weighting
- [graph-based approach](src/graph/readme.md)  
  - Likelihood weighting
  - HMC
- [handler-based approach](src/handler/readme.md)  
  - Likelihood weighting

## Installation
```console
(@v1.8) pkg> add https://github.com/markus7800/TinyPPL.jl
```

```julia
import Pkg
Pkg.add("https://github.com/markus7800/TinyPPL.jl")
```

## Usage

See [examples](examples/readme.md).