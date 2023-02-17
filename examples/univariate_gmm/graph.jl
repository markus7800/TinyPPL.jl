
using TinyPPL.Distributions
using TinyPPL.Graph
import Random
include("common.jl")
mkpath("plots/")

Random.seed!(7800)

println("Compile model")
model = @ppl gmm begin
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


traces, retvals = lmh(model, 100, proposal=Proposal());
@time traces, retvals = lmh(model, 100_000, proposal=Proposal());

println("Get lps")
@time lps = [model.logpdf(traces[:,i]) for i in 1:size(traces,2)]

function to_trace(model, X)
    tr = Dict()
    for i in 1:model.n_variables
        addr = model.addresses[i]
        if addr isa Pair && addr[1] == :z || addr == :k
            tr[addr] = Int(X[i])
        else
            tr[addr] = X[i]
        end
    end
    return tr
end
tr = to_trace(model, traces[:, argmax(lps)])
println("max lp: ", lps[argmax(lps)])
visualize_trace(tr, path="plots/graph_lmh_best_trace.pdf")