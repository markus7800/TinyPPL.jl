
using TinyPPL.Distributions
using TinyPPL.Graph
import Random


model = @pgm normal_chain begin
    let a ~ Normal(0,1),
        b ~ Normal(a+1,1),
        c ~ Normal(b+1,1),
        d ~ Normal(c+1,1),
        e ~ Normal(d+1,1),
        f ~ Normal(e+1,1),
        g ~ Normal(f+1,1),
        h ~ Normal(g+1,1)

        h
    end
end
map_mu = collect(0:7.)
map_sigma = collect(1:8.)

Random.seed!(0)
vi_result = advi_meanfield(model, 10_000, 10, 0.1);
maximum(abs, vi_result.Q.mu .- map_mu)

Random.seed!(0)
@time vi_result = bbvi_naive(model, 10_000, 100, 0.1);
Q_mu = [d.base.μ for d in vi_result.Q.dists]
maximum(abs, map_mu .- Q_mu)

Random.seed!(0)
@time vi_result = bbvi_rao(model, 10_000, 100, 0.01);
Q_mu = [d.base.μ for d in vi_result.Q.dists]
maximum(abs, map_mu .- Q_mu)

import LinearAlgebra: diag
Random.seed!(0)
vi_result = advi_fullrank(model, 100_000, 10, 0.01)
mu, Σ = vi_result.Q.base.μ, vi_result.Q.base.Σ
maximum(abs, mu .- map_mu) # 0.04
maximum(abs, diag(Σ) .- map_sigma) # 0.17
