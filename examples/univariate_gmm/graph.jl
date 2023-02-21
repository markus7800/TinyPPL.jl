using TinyPPL.Graph
import Random

include("common.jl")
mkpath("plots/")

Random.seed!(7800)

println("Compile model")
t0 = time_ns()
model = @ppl plated GMM begin
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

traces, retvals = lmh(model, 100);
@time traces, retvals = lmh(model, 1_000_000);

println("Get lps")
@time lps = [model.logpdf(traces[:,i]) for i in 1:size(traces,2)]

println("Compile LMH")
@time kernels = compile_lmh(model, static_observes=true);
Random.seed!(0);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);
println("Get lps")
@time lps = begin
    X = Vector{Float64}(undef, model.n_variables)
    model.sample(X) # initialises static observed values TODO
    lps = Vector{Float64}(undef, size(traces,2))
    mask = isnothing.(model.observed_values)
    for i in 1:size(traces,2)
        X[mask] = traces[:,i]
        lps[i] = model.logpdf(X)
    end
    lps
end


function to_trace(model, X)
    X_aug = Vector{Float64}(undef, model.n_variables)
    mask = isnothing.(model.observed_values)
    model.sample(X_aug) # initialises static observed values TODO
    X_aug[mask] = X

    tr = Dict()
    for i in 1:model.n_variables
        addr = model.addresses[i]
        if addr isa Pair && addr[1] == :z || addr == :k
            tr[addr] = Int(X_aug[i])
        else
            tr[addr] = X_aug[i]
        end
    end
    return tr
end
tr = to_trace(model, traces[:, argmax(lps)])
println("max lp: ", lps[argmax(lps)])
visualize_trace(tr, path="plots/graph_lmh_best_trace.pdf")