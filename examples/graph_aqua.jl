using TinyPPL.Distributions
using TinyPPL.Graph
import TinyPPL.Graph: Graph

import Distributions as Dists
using Plots


model = @pgm PriorMix begin
    let N = 10,
        y = [1.6024, 0.880643, -0.792556, -1.02184, -0.530578, 0.494777, -1.51075, -0.630318, -1.18208, 0.187967],
        
        b ~ Bernoulli(0.8),
        m = (2 * b - 1) * 4,
        
        # u ~ Uniform(0.,1.),
        # m = (2 * (u < 0.8) - 1) * 4,

        mu ~ Main.Dists.truncated(Normal(m,0.5),-5.,5)

        [{:y => i} ~ TransformedDistribution(TDist(5), AffineTransform(2., mu)) ↦ y[i] for i in 1:N]

        mu
    end
end

@time result = aqua(model, 500, method=:ve);
@time result = aqua(model, 500, method=:bp);
@time result = aqua(model, 500, method=:jt);

xs, x_ps = result[:mu]
plot(xs, x_ps)

xs, x_ps = result[:u]
plot(xs, x_ps, ylim=(0,5))
sum(x_ps[xs .< 0.8]) / sum(x_ps) # u < 0.8 == true
sum(x_ps[xs .> 0.8]) / sum(x_ps) # u < 0.8 == false

xs, x_ps = result[:b]


ys = [1.6024, 0.880643, -0.792556, -1.02184, -0.530578, 0.494777, -1.51075, -0.630318, -1.18208, 0.187967]
mus = LinRange(-5,5,1000)
prior = log.([0.8 * Dists.pdf(Normal(4,0.5), mu) + 0.2 * Dists.pdf(Normal(-4,0.5), mu) for mu in mus])
lik = [(sum(logpdf(TransformedDistribution(TDist(5), AffineTransform(2., mu)), y) for y in ys)) for mu in mus]
#lik = [exp(sum(logpdf(Normal(mu,0.5),y) for y in ys)) for mu in mus]
plot(mus, exp.(prior))
plot(mus, exp.(lik))
plot(mus, exp.(prior .+ lik))

Random.seed!(0)
@time traces, lps = likelihood_weighting(model, 10^6)
histogram(traces[:mu], weights=exp.(lps), normalize=true, legend=false, lc=1)

Random.seed!(0)
@time traces = lmh(model, 10^6, addr2proposal=Addr2Proposal(:mu => StaticProposal(Uniform(-5,5))))#, :u => StaticProposal(Bernoulli(0.2))))
histogram(traces[:mu], normalize=true, legend=false, lc=1)
traces[:u]





model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Normal(X, 1.)

        Normal(Y, 1.) ↦ 3.
    end
end

variable_nodes, factor_nodes = Graph.get_aqua_factor_graph(model)

variable_nodes[1].support
factor_nodes[1].table
sum(exp, factor_nodes[1].table)

v = variable_nodes[2]
f, Z = variable_elimination(model, variable_nodes, factor_nodes, [v.variable], :Greedy)
ps = exp.(f.table)
xs = v.support

Δ = xs[2] - xs[1]
ps = exp.(f.table) / (Z * Δ)
sum(ps) * Δ ≈ 1

plot(xs, ps)


result = aqua_ve(model, 500);

xs, x_ps = result[:X]
ys, y_ps = result[:Y]
plot(xs, x_ps)
plot!(ys, y_ps)


result[:b][2]

Random.seed!(0)
@time traces, lps = likelihood_weighting(model, 10^6)
histogram(traces[:X], weights=exp.(lps), normalize=true, legend=false, lc=1)
plot!(xs, x_ps)
