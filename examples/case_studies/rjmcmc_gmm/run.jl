using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

include("../../common/gmm_data.jl")
include("model.jl")
include("helpers.jl")
include("aux_model.jl")
include("involution.jl")

Random.seed!(0)
@time traces = imcmc(
    gmm, (n,), observations,
    aux_model, (n,),
    involution!,
    100_000 * 6;
    check_involution=true
);
# log p(X,Y) of each trace (MCMC algorithms only return traces)
lp = [score_trace(gmm, (n,), observations, X) for X in traces.data]

# number of times k changes
sum(traces[:k,i] != traces[:k,i+1] for i in 1:(length(traces)-1))

[mean(traces[:k] .== i) for i in 1:10]

maximum(lp)
best_trace = traces.data[argmax(lp)]

include("plotting.jl")

visualize_trace(to_plotting_format(best_trace))
PyPlot.gcf()

for k in sort(unique(traces[:k]))
    mask = traces[:k] .== k
    amax = argmax(lp[mask])
    println("k=$k, lp=$(lp[mask][amax])")
    best_trace_k = traces.data[mask][amax]
    visualize_trace(to_plotting_format(best_trace_k))
    display(PyPlot.gcf())
end

plot_lps(lp[10:end])
PyPlot.gcf()

p = PyPlot.figure()
PyPlot.plot(traces[:k])
PyPlot.gcf()
