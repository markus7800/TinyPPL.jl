# Plates and compiled inference on a Gaussian mixture model with 100 data points (see common/gmm_data.jl).
#
# The model has 112 latent nodes: mixture weights, means and variances of the k = 4 components,
# and a component assignment z_i for each data point. Two ways to speed up graph-based models:
# - The `plated` annotation groups nodes with addresses :y => i (and :z => i) into plates,
#   which makes compiling the model much faster.
# - compile_likelihood_weighting, compile_lmh and compile_rwmh generate specialised code for the model,
#   which takes time to compile but then runs faster. They produce the same samples as the generic algorithms.
# See case_studies/rjmcmc_gmm/ for the same model with an unknown number of components.
#
# Run: julia --project=examples examples/graph/plated_gmm.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import TinyPPL.Evaluation
import Random
include("../common/gmm_data.jl")
include("../common/gmm_plotting.jl")

# The number of components k ~ Poisson(λ - 1) + 1 is observed to be gt_k = 4.
# The Dirichlet distribution of the weights is written as normalised Gamma variables (random variables are univariate).
println("Compiling the model ...")
t_unplated = @elapsed unplated_model = @pgm unplated_GMM begin
    function dirichlet(δ, k)
        let w = [{:w => i} ~ Gamma(δ, 1) for i in 1:k]
            w / sum(w)
        end
    end
    let λ = 3, δ = 5.0, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
        k = ({:k} ~ Poisson(λ - 1) ↦ $(gt_k - 1)) + 1,
        y = $(gt_ys),
        n = length(y),
        w = dirichlet(δ, k),
        means = [{:μ => j} ~ Normal(ξ, 1 / sqrt(κ)) for j in 1:k],
        vars = [{:σ² => j} ~ InverseGamma(α, β) for j in 1:k],
        z = [{:z => i} ~ Categorical(w) for i in 1:n]

        [{:y => i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]

        means
    end
end

t_plated = @elapsed model = @pgm plated plated_GMM begin
    function dirichlet(δ, k)
        let w = [{:w => i} ~ Gamma(δ, 1) for i in 1:k]
            w / sum(w)
        end
    end
    let λ = 3, δ = 5.0, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
        k = ({:k} ~ Poisson(λ - 1) ↦ $(gt_k - 1)) + 1,
        y = $(gt_ys),
        n = length(y),
        w = dirichlet(δ, k),
        means = [{:μ => j} ~ Normal(ξ, 1 / sqrt(κ)) for j in 1:k],
        vars = [{:σ² => j} ~ InverseGamma(α, β) for j in 1:k],
        z = [{:z => i} ~ Categorical(w) for i in 1:n]

        [{:y => i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]

        means
    end
end
println("compilation time: unplated ", round(t_unplated, digits=1), "s, plated ", round(t_plated, digits=1), "s (", model.n_latents, " latent nodes)")
println("plates: ", model.plate_info.plates)


# ===== Compiled inference =====
# Each pair runs the generic and the compiled algorithm with the same seed, which give the same samples.
# The first call of each algorithm also compiles Julia code, so we run each once before timing.
n_samples = 100_000
logjoint(traces) = [model.logpdf(traces[:, i], model.observations) for i in 1:length(traces)]
function compare(name, generic, compile, compiled; result=traces -> traces.data)
    generic(10)
    Random.seed!(0)
    t_generic = @elapsed generic_output = generic(n_samples)
    t_compile = @elapsed compiled_algorithm = compile()
    compiled(compiled_algorithm, 10)
    Random.seed!(0)
    t_compiled = @elapsed compiled_output = compiled(compiled_algorithm, n_samples)
    println(rpad(name, 22), "generic ", round(t_generic, digits=2), "s, compiled ", round(t_compiled, digits=2),
        "s (+ ", round(t_compile, digits=1), "s compilation), same samples: ", result(generic_output) == result(compiled_output))
    return compiled_output
end

println()
# Likelihood weighting is a poor algorithm for this model: its log evidence estimates vary by more than 100
# between seeds, as a single sample dominates the weights.
compare("likelihood weighting",
    n -> likelihood_weighting(model, n),
    () -> compile_likelihood_weighting(model),
    (lw, n) -> compiled_likelihood_weighting(model, lw, n),
    result=output -> output[1].data)
compare("LMH",
    n -> lmh(model, n),
    () -> compile_lmh(model),
    (kernels, n) -> compiled_single_site(model, kernels, n))
# Random-walk proposals with a variance per address. For the discrete z, a random walk always moves away
# from the current value, which gives a low acceptance rate.
addr2var = Addr2Var(:μ => 0.5, :σ² => 2., :w => 5., :z => 1000.)
rwmh_traces = compare("RWMH",
    n -> rwmh(model, n, addr2var=addr2var),
    () -> compile_rwmh(model, addr2var=addr2var),
    (kernels, n) -> compiled_single_site(model, kernels, n))

# The most likely sample of the RWMH chain, together with the ground truth components (gray).
lps = logjoint(rwmh_traces)
X = rwmh_traces[:, argmax(lps)]
trace = Dict{Any,Real}(model.addresses[node] => X[node] for node in 1:model.n_latents)
trace[:k] = gt_k
w_sum = sum(trace[:w => j] for j in 1:gt_k)
for j in 1:gt_k
    trace[:w => j] /= w_sum
end
println("\nmost likely RWMH sample: means ", sort([round(trace[:μ => j], digits=2) for j in 1:gt_k]), " (ground truth: ", gt_μs, ")")
show_plot(visualize_trace(trace), "graph_plated_gmm")


# ===== Comparison with an evaluation-based model =====
# The same model as universal model is interpreted instead of compiled, so LMH is much slower.
const λ, δ, ξ, κ, α, β = 3, 5.0, 0.0, 0.01, 2.0, 10.0

Evaluation.@ppl function gmm(n)
    k = {:k} ~ Poisson(λ - 1)
    k = k + 1
    w = [{:w => j} ~ Gamma(δ, 1) for j in 1:k]
    w = w / sum(w)
    means, vars = zeros(k), zeros(k)
    for j in 1:k
        means[j] = {:μ => j} ~ Normal(ξ, 1 / sqrt(κ))
        vars[j] = {:σ² => j} ~ InverseGamma(α, β)
    end
    for i in 1:n
        z = {:z => i} ~ Categorical(w)
        {:y => i} ~ Normal(means[z], sqrt(vars[z]))
    end
end

observations = Evaluation.Observations((:y => i) => y for (i, y) in enumerate(gt_ys))
observations[:k] = gt_k - 1
Evaluation.lmh(gmm, (length(gt_ys),), observations, 10)
Random.seed!(0)
t = @elapsed Evaluation.lmh(gmm, (length(gt_ys),), observations, n_samples)
println("\nLMH on the evaluation-based model: ", round(t, digits=2), "s")
