TODOS:

- speed up and fix junction tree (like greedy variable elimination)
- document variable elimination belief propagation etc
- belief propagation with queue instead of recursion
- reduce compile time in graph by batching distributions, observed_values functions similar to plated

- AQUA inference for graphsample
- 
- document everyting


- rjmcmc
- smc (graph ok / universial with channels?)
- logjoint -> logdensity
- graph -> enfore static observes

- universal
- IS Proposal and Guide
- MH with Guide

TinyGen:
- one sampler:
  - sample / or logpdf:
    - get!(sampler.trace, addr, rand(dist))
    - get!(sampler.trace, addr, Tracker.param(rand(dist)))
  - param: 
    - get!(sampler.params, addr, Tracker.param(zero(size)))
- ForwardDiff Dictionaries.jl jacobian

- assess
- simulate
- propose
- gradients
- etc...