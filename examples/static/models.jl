# Introduction to static models.
#
# `@ppl static function` declares that the model instantiates the same finite set of addresses in every run
# (for fixed arguments). Then the addresses can be collected once by running the model,
# and traces are stored as vectors (one entry per address) instead of dictionaries.
# This makes inference more efficient and enables vector-based algorithms like HMC, ADVI and SMC,
# see the other examples in this folder.
#
# Run: julia --project=examples examples/static/models.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

# A Gaussian random walk of N steps, whose last position is observed with noise.
# The loop runs a fixed number of times, so the model is static.
@ppl static function random_walk(N)
    X = 0.
    for i in 1:N
        X = {:X => i} ~ Normal(X, 1.)
    end
    Z ~ Normal(X, 1.)
    return X
end

N = 100
z = 3.
args = (N,)
observations = Observations(:Z => z)

# Each latent address is mapped to an index into the trace vector.
addresses_to_ix = get_address_to_ix(random_walk, args, observations)
println("Number of latent variables: ", length(addresses_to_ix), ", index of :X => N: ", addresses_to_ix[:X => N])

# Likelihood weighting returns the traces, the normalised log weights, and an estimate of the log evidence log p(Z).
Random.seed!(0)
traces, logw, log_Z = likelihood_weighting(random_walk, args, observations, 100_000)
W = exp.(logw)

# Traces store a matrix with one column per sample, which can be indexed by address and sample index
# (`traces[addr]` for all samples, `traces[addr, i]` for one, `traces[addr, ixs]` for a range, index vector or mask).
println("Stored trace matrix: ", size(traces.data))
println("First 3 samples of X => N: ", traces[:X => N, 1:3], ", return values: ", retvals(traces)[1:3])
# The model is jointly Gaussian: a priori X_N ~ Normal(0, N), so
# p(X_N | Z = z) = Normal(N / (N+1) * z, N / (N+1)) and p(Z = z) = Normal(z; 0, N+1).
println("Likelihood weighting: E[X_N | Z] ≈ ", traces[:X => N]'W, " (exact: ", N / (N+1) * z, ")")
println("Likelihood weighting: log p(Z) ≈ ", log_Z, " (exact: ", logpdf(Normal(0., sqrt(N+1)), z), ")")

# If we are only interested in some addresses, we can pass them to `addresses` and only their values are stored.
# With the same seed, the results are identical.
Random.seed!(0)
traces_X_N, logw, log_Z = likelihood_weighting(random_walk, args, observations, 100_000; addresses=Address[:X => N])
println("Stored trace matrix with addresses=[:X => N]: ", size(traces_X_N.data))
println("Same values as before: ", traces_X_N[:X => N] == traces[:X => N])
