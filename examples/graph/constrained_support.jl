# Variables whose support depends on other variables, as graph-based model.
#
# x ~ Uniform(-1, 1), y ~ Uniform(x - 1, x + 1), z ~ Uniform(y - 1, y + 1), nothing observed, so y has the triangular
# density (2 - |y|) / 4 on [-2, 2], std(y) = sqrt(2/3), std(z) = 1, and P(|y| < 1) = 0.75.
# Each variable is mapped to ℝ with a bijection that depends on its parents' values.
# See static/constrained_support.jl for a more detailed explanation.
#
# Run: julia --project=examples examples/graph/constrained_support.jl

using TinyPPL.Distributions
using TinyPPL.Graph
import Random
using StatsPlots
include("../common/plotting.jl")

model = @pgm uniform_chain begin
    let x ~ Uniform(-1., 1.),
        y ~ Uniform(x - 1., x + 1.),
        z ~ Uniform(y - 1., y + 1.)

        z
    end
end

_std(v) = sqrt(mean((v .- mean(v)) .^ 2))
print_summary(method, traces) = println(rpad(method, 24), "std(y) ≈ ", round(_std(traces[:y]), digits=3),
    ", std(z) ≈ ", round(_std(traces[:z]), digits=3), ", P(|y| < 1) ≈ ", round(mean(abs.(traces[:y]) .< 1), digits=3))
println(rpad("exact", 24), "std(y) = ", round(sqrt(2/3), digits=3), ", std(z) = 1.0, P(|y| < 1) = 0.75")

Random.seed!(0)
hmc_traces = hmc(model, 100_000, 10, 0.1; unconstrained=true)
print_summary("HMC", hmc_traces)

Random.seed!(0)
vi_result = advi_meanfield(model, 10_000, 10, 0.01)
advi_traces = sample_posterior(vi_result, 100_000)
print_summary("ADVI mean-field", advi_traces)

Random.seed!(0)
vi_result = bbvi_rao(model, 10_000, 10, 0.01)
bbvi_traces = sample_posterior(vi_result, 100_000)
print_summary("BBVI Rao-Blackwellised", bbvi_traces)

p = histogram(hmc_traces[:y], normalize=true, label="HMC", linewidth=0, alpha=0.5, xlabel="y")
histogram!(p, advi_traces[:y], normalize=true, label="ADVI", linewidth=0, alpha=0.5)
histogram!(p, bbvi_traces[:y], normalize=true, label="BBVI", linewidth=0, alpha=0.5)
plot!(p, -2:0.01:2, y -> (2 - abs(y)) / 4, label="exact", color=:black)
show_plot(p, "graph_constrained_support")
