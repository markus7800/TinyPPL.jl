using TinyPPL.Distributions
using TinyPPL.Graph
using TinyPPL.Evaluation
import Random

include("gmm/data.jl")
include("gmm/common.jl")

@time begin
    println("Compile model")
    t0 = time_ns()
    unplated_model = @pgm unplated_GMM begin
        function dirichlet(δ, k)
            let w = [{:w=>i} ~ Gamma(δ, 1) for i in 1:k]
                w / sum(w)
            end
        end
        let λ = 3, δ = 5.0, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
            k = ({:k} ~ Poisson(λ) ↦ 3) + 1,
            y = $(Main.gt_ys),
            n = length(y),
            w = dirichlet(δ, k),
            means = [{:μ=>j} ~ Normal(ξ, 1/sqrt(κ)) for j in 1:k],
            vars = [{:σ²=>j} ~ InverseGamma(α, β) for j in 1:k],
            z = [{:z=>i} ~ Categorical(w) for i in 1:n]

            [{:y=>i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]
            
            means
        end
    end
    println("Compilation Time: ", (time_ns() - t0) / 1e9)
end


@time begin
    println("Compile plated model")
    t0 = time_ns()
    plated_model = @pgm plated plated_GMM begin
        function dirichlet(δ, k)
            let w = [{:w=>i} ~ Gamma(δ, 1) for i in 1:k]
                w / sum(w)
            end
        end
        let λ = 3, δ = 5.0, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
            k = ({:k} ~ Poisson(λ) ↦ 3) + 1,
            y = $(Main.gt_ys),
            n = length(y),
            w = dirichlet(δ, k),
            means = [{:μ=>j} ~ Normal(ξ, 1/sqrt(κ)) for j in 1:k],
            vars = [{:σ²=>j} ~ InverseGamma(α, β) for j in 1:k],
            z = [{:z=>i} ~ Categorical(w) for i in 1:n]

            [{:y=>i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]
            
            means
        end
    end
    println("Compilation Time: ", (time_ns() - t0) / 1e9)
end
function graph_to_plotting_format(X)
    d = Dict{Any,Real}(model.addresses[n] => X[n] for n in 1:model.n_latents)
    d[:k] = gt_k
    w_sum = sum(d[:w => k] for k in 1:gt_k)
    for k in 1:gt_k
        d[:w => k] = d[:w => k] / w_sum
    end
    return d
end

# model = unplated_model;
model = plated_model;

@info "LW"
_ = Graph.likelihood_weighting(model, 100);
Random.seed!(0); @time traces1, lps1 = Graph.likelihood_weighting(model, 100_000);

println("Compile LW")
@time lw = compile_likelihood_weighting(model)
_ = Graph.compiled_likelihood_weighting(model, lw, 100);
Random.seed!(0); @time traces2, lps2 = compiled_likelihood_weighting(model, lw, 100_000);

lps1 ≈ lps2



@info "LMH"
_ = Graph.lmh(model, 100);
Random.seed!(0); @time traces1 = Graph.lmh(model, 100_000);

println("Get lps")
@time lps1 = [model.logpdf(traces1[:,i], model.observations) for i in 1:length(traces1)];

println("Compile LMH")
@time kernels = compile_lmh(model);
Random.seed!(0); @time traces2 = compiled_single_site(model, kernels, 100_000);
println("Get lps")
@time lps2 = [model.logpdf(traces2[:,i], model.observations) for i in 1:length(traces2)];

lps1 ≈ lps2


@info "RWMH"
addr2var = Addr2Var(:μ=>0.5, :σ²=>2., :w=>5., :z=>1000.)
_ = Graph.rwmh(model, 100, addr2var=addr2var);
# acceptance rate for z is much worse because we force a move / don't stay at current value
Random.seed!(0); @time traces1 = Graph.rwmh(model, 100_000, addr2var=addr2var);
println("Get lps")
@time lps1 = [model.logpdf(traces1[:,i], model.observations) for i in 1:length(traces1)];

println("Compile RMWH")
@time kernels = compile_rwmh(model, addr2var=addr2var);
Random.seed!(0); @time traces2 = compiled_single_site(model, kernels, 100_000);
println("Get lps")
@time lps2 = [model.logpdf(traces2[:,i], model.observations) for i in 1:length(traces2)];

lps1 ≈ lps2

maximum(lps2)
X = traces2[:, argmax(lps2)]
visualize_trace(graph_to_plotting_format(X))

const λ = 3
const δ = 5.0
const ξ = 0.0
const κ = 0.01
const α = 2.0
const β = 10.0

@ppl function dirichlet(δ, k)
    w = [{:w=>j} ~ Gamma(δ, 1) for j in 1:k]
    return w / sum(w)
end

@ppl function gmm(n)
    k = {:k} ~ Poisson(λ)
    k = k + 1
    w = @subppl dirichlet(δ, k)

    means, vars = zeros(k), zeros(k)
    for j=1:k
        means[j] = ({:μ=>j} ~ Normal(ξ, 1/sqrt(κ)))
        vars[j] = ({:σ²=>j} ~ InverseGamma(α, β))
    end
    for i=1:n
        z = {:z=>i} ~ Categorical(w)
        z > k && continue # then trace has logpdf -Inf anyways
        {:y=>i} ~ Normal(means[z], sqrt(vars[z]))
    end
end

function universal_to_plotting_format(X)
    d = copy(X)
    d[:k] = gt_k
    w_sum = sum(d[:w => k] for k in 1:gt_k)
    for k in 1:gt_k
        d[:w => k] = d[:w => k] / w_sum
    end
    return d
end

args = (length(gt_ys), )

observations = Observations((:y=>i)=>y for (i, y) in enumerate(gt_ys))
observations[:k] = gt_k-1

@info("LW")
Random.seed!(0); @time traces, lps = Evaluation.likelihood_weighting(gmm, (length(gt_ys),), observations, 100_000);


@info("LMH")
_ = Evaluation.lmh(gmm, args, observations, 100);
Random.seed!(0); @time traces = Evaluation.lmh(gmm, (length(gt_ys),), observations, 100_000);


@info("RWMH")
_ = Evaluation.rwmh(gmm, (length(gt_ys), ), observations, 100);
# acceptance rate for z is much worse because we force a move / don't stay at current value
Random.seed!(0); @time traces = Evaluation.rwmh(gmm, args, observations, 100_000, addr2var=Addr2Var(:μ=>0.5, :σ²=>2., :w=>5., :z=>1000.));

logjoint = Evaluation.make_logjoint(gmm, args, observations)
X = argmax(logjoint, traces.data)
logjoint(X)
visualize_trace(universal_to_plotting_format(X))
