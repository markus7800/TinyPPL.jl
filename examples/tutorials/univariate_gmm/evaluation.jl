
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
include("data.jl")
include("common.jl")
mkpath("plots/")

Random.seed!(7800)


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

const observations = Dict{Any, Real}((:y=>i)=>y for (i, y) in enumerate(gt_ys))
observations[:k] = gt_k-1
# for i in 1:length(gt_ys)
#     observations[(:z=>i)] = gt_zs[i]
# end

# sampler = Forward();
# @info("Forward")
# [gmm(length(gt_ys), sampler, observations) for _ in 1:100];
# @time [gmm(length(gt_ys), sampler, observations) for _ in 1:1_000_000];

@info("LMH")
traces, retvals, lps = lmh(gmm, (length(gt_ys), ), observations, 100, proposal=Proposal());
@time traces, retvals, lps = lmh(gmm, (length(gt_ys),), observations, 100_000, proposal=Proposal());
tr = traces[argmax(lps)]
tr = merge(tr, observations)
# plot_lps(lps, path="plots/lmh_lps.pdf")
println("max lp: ", lps[argmax(lps)])
visualize_trace(tr, path="plots/evaluation_lmh_best_trace.pdf")


k = tr[:k]+1
w_sum = sum(tr[:w=>j] for j in 1:k)
for j in 1:k
    println("μ$j: ", tr[:μ=>j], " with var ", tr[:σ²=>j], " and w ", tr[:w=>j] / w_sum)
end
# means = [tr[(:μ=>j)] for j in 1:k, tr in traces[10_000:end]]
# vars = [tr[(:σ²=>j)] for j in 1:k, tr in traces[10_000:end]]
# ws = [tr[(:w=>j)] for j in 1:k, tr in traces[10_000:end]]
# zs = [tr[(:z=>i)] for i in 1:length(gt_ys), tr in traces[10_000:end]]
# import Distributions
# println("mean vars: ", Distributions.var(means, dims=2))
# println("vars vars: ", Distributions.var(vars, dims=2))
# println("ws vars: ", Distributions.var(ws, dims=2))
# println("zs vars: ", Distributions.var(zs[1:10,:], dims=2))

# plot_params(means, path="plots/lmh_means.pdf")
# plot_params(vars, path="plots/lmh_vars.pdf")
# plot_params(ws, path="plots/lmh_ws.pdf")
# plot_params(zs[1:3,:], path="plots/lmh_zs.pdf")

traces, retvals, lps = rwmh(gmm, (length(gt_ys), ), observations, 100);
# acceptance rate for z is much worse because we force a move / don't stay at current value
@time traces, retvals, lps = rwmh(gmm, (length(gt_ys),), observations, 100_000, addr2var=Addr2Var(:μ=>0.5, :σ²=>2., :w=>5., :z=>1000.));
tr = traces[argmax(lps)]
tr = merge(tr, observations)
# plot_lps(lps, path="plots/rwmh_lps.pdf")
println("max lp: ", lps[argmax(lps)])
visualize_trace(tr, path="plots/evaluation_rwmh_best_trace.pdf")
# display(tr)

# means = [tr[(:μ=>j)] for j in 1:k, tr in traces]
# vars = [tr[(:σ²=>j)] for j in 1:k, tr in traces]
# ws = [tr[(:w=>j)] for j in 1:k, tr in traces]
# zs = [tr[(:z=>i)] for i in 1:length(gt_ys), tr in traces[10_000:end]]

# println("mean vars: ", Distributions.var(means, dims=2))
# println("vars vars: ", Distributions.var(vars, dims=2))
# println("ws vars: ", Distributions.var(ws, dims=2))
# println("zs vars: ", Distributions.var(zs[1:10,:], dims=2))

# plot_params(means, path="plots/rwmh_means.pdf")
# plot_params(vars, path="plots/rwmh_vars.pdf")
# plot_params(ws, path="plots/rwmh_ws.pdf")
# plot_params(zs[1:3,:], path="plots/rwmh_zs.pdf")

k = tr[:k]+1
w_sum = sum(tr[:w=>j] for j in 1:k)
for j in 1:k
    println("μ$j: ", tr[:μ=>j], " with var ", tr[:σ²=>j], " and w ", tr[:w=>j] / w_sum)
end
# display(tr)

# println([sum(tr[:k] == i for tr in traces) for i in 0:10])
