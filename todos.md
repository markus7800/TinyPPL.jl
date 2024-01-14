TODOS:

- speed up and fix junction tree (like greedy variable elimination)
- document variable elimination belief propagation etc
- belief propagation with queue instead of recursion
- reduce compile time in graph by batching distributions, observed_values functions similar to plated

- graph: TODO: x[1] = 1 not allowed, immutable, but y = x[1] is allowed

- AQUA inference for graphsample
- 
- document everyting


- fix func(; a = ..) keyword arguments @ppl eval macro
- param transform reuse transform.jl? transform_to RealInterval
- remove observations from guide? can be passed as argument instead
- HMC find eps ala mcmcgallery

- param syntax: p = {:p => i} ∈ ℝ^n = init (0,∞)^n etc \bbR

- rjmcmc
- smc (graph ok / universial with channels?)
- logjoint -> logdensity
- graph -> enfore static observes

- universal
- IS Proposal and Guide
- MH with Guide
- Nonparametric HMC

- model learning -> maximising log p(Y)

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