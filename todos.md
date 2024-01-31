TODOS:

- belief propagation with queue instead of recursion
- reduce compile time in graph by batching distributions, observed_values functions similar to plated

- graph: TODO: x[1] = 1 not allowed, immutable, but y = x[1] is allowed

- AQUA inference for graphsample
- 
- truncate output of pgm if too long
- fix: is_tree should be indicative of whether we can construct belief tree see bnlearn networks (e.g. pigs)

- understand HMC with discontinuities see non-parametric hmc efforts. / mass matrix / make golf work

- fix func(; a = ..) keyword arguments @ppl eval macro
- param transform reuse transform.jl? transform_to RealInterval
- remove observations from guide? can be passed as argument instead
- HMC find eps ala mcmcgallery

- param syntax: p = {:p => i} ∈ ℝ^n = init (0,∞)^n etc \bbR

- logjoint -> logdensity

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