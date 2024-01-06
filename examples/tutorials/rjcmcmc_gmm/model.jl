# ===== model ===========================================================================

const λ = 3
const δ = 5.0
const ξ = 0.0
const κ = 0.01
const α = 2.0
const β = 10.0

@ppl function dirichlet(δ, k)
    w = zeros(k)
    α0 = δ * k
    s = 0 # sum_{i=1}^{j-1} w[i]
    for j in 1:k-1
        α0 -= δ
        phi = {:phi => j} ~ Beta(δ, α0)
        w[j] = (1-s) * phi
        s += w[j]
    end
    w[k] = 1 - s
    return w
end

import Distributions
struct PositivePoisson <: Distributions.DiscreteUnivariateDistribution
    λ
end
Distributions.rand(d::PositivePoisson) = Distributions.rand(Poisson(d.λ)) + 1
Distributions.logpdf(d::PositivePoisson, x) = Distributions.logpdf(Poisson(d.λ), x-1)

@ppl function gmm(n::Int)
    k = {:k} ~ PositivePoisson(λ)
    w = @subppl dirichlet(δ, k)

    means, vars = zeros(k), zeros(k)
    for j=1:k
        means[j] = ({:μ=>j} ~ Normal(ξ, 1/sqrt(κ)))
        vars[j] = ({:σ²=>j} ~ InverseGamma(α, β))
    end
    for i=1:n
        z = {:z=>i} ~ Categorical(w)
        {:y=>i} ~ Normal(means[z], sqrt(vars[z]))
    end
end

const observations = Observations((:y=>i)=>y for (i, y) in enumerate(gt_ys))