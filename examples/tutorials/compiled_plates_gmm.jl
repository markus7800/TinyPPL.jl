using TinyPPL.Graph
using TinyPPL.Distributions
import Distributions: mean, std
import Random

include("gmm/data.jl")

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

model = unplated_model;
model = plated_model;

@info "LW"
traces, lps = likelihood_weighting(model, 100);
@time traces, lps = likelihood_weighting(model, 100_000);

println("Compile LW")
@time lw = compile_likelihood_weighting(model)
traces, lps = compiled_likelihood_weighting(model, lw, 100);
@time traces, lps = compiled_likelihood_weighting(model, lw, 1_000_000);




@info "LMH"
traces = lmh(model, 100);
Random.seed!(0); @time traces = lmh(model, 1_000_000);
mean(traces.data, dims=2)

println("Get lps")
@time lps = [model.logpdf(traces[:,i], model.observations) for i in 1:length(traces)];

println("Compile LMH")
@time kernels = compile_lmh(model);
Random.seed!(0); @time traces2 = compiled_single_site(model, kernels, 1_000_000);
println("Get lps")
@time lps2 = [model.logpdf(traces2[:,i], model.observations) for i in 1:length(traces2)];


@info "RWMH"
addr2var = Addr2Var(:μ=>0.5, :σ²=>2., :w=>5., :z=>1000.)
traces = rwmh(model, 100, addr2var=addr2var);
# acceptance rate for z is much worse because we force a move / don't stay at current value
Random.seed!(0); @time traces = rwmh(model, 1_000_000, addr2var=addr2var);
println("Get lps")
@time lps = [model.logpdf(traces[:,i], model.observations) for i in 1:length(traces)];

println("Compile RMWH")
@time kernels = compile_rwmh(model, addr2var=addr2var);
Random.seed!(0); @time traces2 = compiled_single_site(model, kernels, 1_000_000);
println("Get lps")
@time lps2 = [model.logpdf(traces2[:,i], model.observations) for i in 1:length(traces2)];