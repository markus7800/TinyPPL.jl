using TinyPPL.Graph
import Random
include("data.jl")
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

function to_trace(model, X)
    if length(X) < model.n_variables
        X_aug = Vector{Float64}(undef, model.n_variables)
        mask = isnothing.(model.observed_values)
        model.sample(X_aug) # initialises static observed values TODO
        X_aug[mask] = X
    else
        X_aug = X
    end

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

function get_lps_from_static_observes(traces)
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

@info "LW"
traces, retvals, lps = likelihood_weighting(model, 100);
@time traces, retvals, lps = likelihood_weighting(model, 100_000);

println("Compile LW")
@time lw = compile_likelihood_weighting(model)
traces, retvals, lps = compiled_likelihood_weighting(model, lw, 100; static_observes=true);
@time traces, retvals, lps = compiled_likelihood_weighting(model, lw, 1_000_000; static_observes=true);


@info "LMH"
traces, retvals = lmh(model, 100);
@time traces, retvals = lmh(model, 1_000_000);

println("Get lps")
@time lps = [model.logpdf(traces[:,i]) for i in 1:size(traces,2)]

println("Compile LMH")
@time kernels = compile_lmh(model, static_observes=true);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);
println("Get lps")
@time lps = get_lps_from_static_observes(traces)

tr = to_trace(model, traces[:, argmax(lps)])
println("max lp: ", lps[argmax(lps)])
visualize_trace(tr, path="plots/graph_lmh_best_trace.pdf")

@info "RWMH"
const addr2var = Addr2Var(:μ=>0.5, :σ²=>2., :w=>5., :z=>1000.)
traces, retvals = rwmh(model, 100, addr2var=addr2var);
# acceptance rate for z is much worse because we force a move / don't stay at current value
@time traces, retvals = rwmh(model, 1_000_000, addr2var=addr2var);
println("Get lps")
@time lps = [model.logpdf(traces[:,i]) for i in 1:size(traces,2)]

println("Compile RMWH")
@time kernels = compile_rwmh(model, static_observes=true, addr2var=addr2var);
@time traces, retvals = compiled_single_site(model, kernels, 1_000_000, static_observes=true);
@time lps = get_lps_from_static_observes(traces)

tr = to_trace(model, traces[:, argmax(lps)])
println("max lp: ", lps[argmax(lps)])
visualize_trace(tr, path="plots/graph_rwmh_best_trace.pdf")