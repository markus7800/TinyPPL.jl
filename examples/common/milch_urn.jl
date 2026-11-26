# Data and exact posterior for the urn example (universal/milch_urn.jl, graph/milch_urn.jl)
# from Milch et al., Approximate Inference for Infinite Contingent Bayesian Networks (AISTATS 2005).
#
# An urn contains an unknown number of balls N ~ Poisson(6), each black or white with probability 0.5.
# We draw K = 10 times with replacement and see 5 black and 5 white balls. How many balls are in the urn?
#
# As in Milch et al., each observed colour is wrong with probability `noise` = 0.2 by default.
# Pass the noise as argument to the scripts, e.g. 0 for noise-free observations:
#   julia --project=examples examples/universal/milch_urn.jl 0

import Distributions: Binomial, Poisson, logpdf
import TinyPPL: logsumexp

const noise = isempty(ARGS) ? 0.2 : parse(Float64, ARGS[1])
const K = 10
const colors = [isodd(k) ? 1. : 0. for k in 1:K] # observed colours: 1 = black, 0 = white
const n_black_observed = Int(sum(colors))
const N_max = 15

# log p(colors | N): given N, the number of black balls n is Binomial(N, 0.5),
# each draw is black with probability n / N, and is observed as black with probability r below.
function log_likelihood(N::Int)
    N == 0 && return -Inf # no ball can be drawn from an empty urn
    return logsumexp([
        let r = n / N * (1 - noise) + (1 - n / N) * noise
            logpdf(Binomial(N, 0.5), n) + n_black_observed * log(r) + (K - n_black_observed) * log(1 - r)
        end
        for n in 0:N
    ])
end

# Posterior over N = 0, ..., N_max (by default, P(N > N_max) ≈ 1e-4 a priori is ignored).
function exact_posterior(N_max::Int=N_max)
    log_joint = [logpdf(Poisson(6), N) + log_likelihood(N) for N in 0:N_max]
    return exp.(log_joint .- logsumexp(log_joint))
end
