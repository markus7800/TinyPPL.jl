#=
Reference posteriors for the test models.

Every model used in the tests is defined (mathematically) here once, together with
its exact posterior quantities. They are computed in closed form, by enumeration,
or by 1-D quadrature — never with TinyPPL itself.

The PPL-specific model definitions live in the individual test files and must
match the descriptions below.
=#
module References

import Distributions
import Distributions: Normal, Gamma, Beta, Bernoulli, Categorical, Poisson, MvNormal, logpdf, pdf
import SpecialFunctions: logbeta, loggamma
import LinearAlgebra: I, diag

logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : m + log(sum(exp, x .- m)))

# ------------------------------------------------------------------------------
# M1 normal_normal: continuous, unconstrained, conjugate
#   μ ~ Normal(0, 2)
#   y[i] ~ Normal(μ, 1)  for y = [1.0, 2.0, 1.5]
# ------------------------------------------------------------------------------
const NN_PRIOR_SD = 2.0
const NN_YS = [1.0, 2.0, 1.5]
const NN_POST_VAR = 1 / (1 / NN_PRIOR_SD^2 + length(NN_YS))
const NN_POST_MEAN = NN_POST_VAR * sum(NN_YS)
const NN_POST_SD = sqrt(NN_POST_VAR)
const NN_LOG_EVIDENCE = logpdf(MvNormal(zeros(3), NN_PRIOR_SD^2 * ones(3, 3) + I), NN_YS)

# ------------------------------------------------------------------------------
# M2 gamma_scale: continuous, positive latent which is the *parent* (scale) of observations
#   s ~ Gamma(2, 1)            (shape, scale)
#   y[i] ~ Normal(0, s)  for y = [0.5, -1.2, 2.0]
# reference by quadrature
# ------------------------------------------------------------------------------
const GS_YS = [0.5, -1.2, 2.0]
function _gamma_scale_quadrature()
    grid = range(1e-6, 40.0, length=400_001)
    Δ = step(grid)
    logp = [logpdf(Gamma(2.0, 1.0), s) + sum(logpdf(Normal(0.0, s), y) for y in GS_YS) for s in grid]
    log_Z = logsumexp(logp) + log(Δ)
    w = exp.(logp .- log_Z) .* Δ
    m = sum(w .* grid)
    v = sum(w .* (grid .- m) .^ 2)
    return m, sqrt(v), log_Z
end
const GS_POST_MEAN, GS_POST_SD, GS_LOG_EVIDENCE = _gamma_scale_quadrature()

# ------------------------------------------------------------------------------
# M3 beta_bernoulli: continuous latent on (0,1), conjugate
#   p ~ Beta(2, 2)
#   y[i] ~ Bernoulli(p)  for y = [1, 1, 0, 1, 1]
# ------------------------------------------------------------------------------
const BB_YS = [1.0, 1.0, 0.0, 1.0, 1.0]
const BB_POST = Beta(2 + sum(BB_YS), 2 + length(BB_YS) - sum(BB_YS))
const BB_POST_MEAN = Distributions.mean(BB_POST)
const BB_POST_SD = Distributions.std(BB_POST)
const BB_LOG_EVIDENCE = logbeta(BB_POST.α, BB_POST.β) - logbeta(2, 2)

# ------------------------------------------------------------------------------
# M4 discrete_network: discrete only, tree-structured
#   z ~ Categorical([0.5, 0.3, 0.2])
#   b ~ Bernoulli(z == 1 ? 0.1 : (z == 2 ? 0.5 : 0.9))
#   o1 ~ Bernoulli(b == 1 ? 0.8 : 0.3)   observed 1
#   o2 ~ Bernoulli(z == 3 ? 0.7 : 0.2)   observed 0
# reference by enumeration
# ------------------------------------------------------------------------------
dn_pb(z) = z == 1 ? 0.1 : (z == 2 ? 0.5 : 0.9)
dn_po1(b) = b == 1 ? 0.8 : 0.3
dn_po2(z) = z == 3 ? 0.7 : 0.2
function _discrete_network_enumeration()
    joint = zeros(3, 2) # z, b+1
    for z in 1:3, b in 0:1
        joint[z, b+1] = pdf(Categorical([0.5, 0.3, 0.2]), z) *
                        pdf(Bernoulli(dn_pb(z)), b) *
                        pdf(Bernoulli(dn_po1(b)), 1) *
                        pdf(Bernoulli(dn_po2(z)), 0)
    end
    Z = sum(joint)
    joint ./= Z
    return vec(sum(joint, dims=2)), vec(sum(joint, dims=1)), log(Z)
end
const DN_POST_Z, DN_POST_B, DN_LOG_EVIDENCE = _discrete_network_enumeration() # P(z=1..3), P(b=0..1)

# ------------------------------------------------------------------------------
# M5 mixed: discrete parent, continuous child
#   z ~ Bernoulli(0.3)
#   x ~ Normal(z == 1 ? 2.0 : -1.0, 1.0)
#   y ~ Normal(x, 0.5)   observed 0.8
# ------------------------------------------------------------------------------
const MX_Y = 0.8
mx_mean(z) = z == 1 ? 2.0 : -1.0
function _mixed_reference()
    # y | z ~ Normal(m_z, sqrt(1 + 0.25))
    lp = [log(0.7) + logpdf(Normal(mx_mean(0), sqrt(1.25)), MX_Y),
          log(0.3) + logpdf(Normal(mx_mean(1), sqrt(1.25)), MX_Y)]
    log_Z = logsumexp(lp)
    p_z1 = exp(lp[2] - log_Z)
    # x | z, y: precision 1 + 4, mean (m_z + 4y) / 5
    ex = [(mx_mean(z) + 4 * MX_Y) / 5 for z in 0:1]
    post_mean_x = (1 - p_z1) * ex[1] + p_z1 * ex[2]
    second_moment = (1 - p_z1) * (1/5 + ex[1]^2) + p_z1 * (1/5 + ex[2]^2)
    return p_z1, post_mean_x, sqrt(second_moment - post_mean_x^2), log_Z
end
const MX_POST_Z1, MX_POST_MEAN_X, MX_POST_SD_X, MX_LOG_EVIDENCE = _mixed_reference()

# ------------------------------------------------------------------------------
# M6 chain: continuous chain whose address order (:a < :o < :z) differs from topological order
#   z ~ Normal(0, 1)
#   a ~ Normal(z, 0.5)
#   o ~ Normal(a, 1)   observed 1.0
# Var(a) = 1.25, Var(o) = 2.25, Cov(z, o) = 1, Cov(a, o) = 1.25
# ------------------------------------------------------------------------------
const CH_O = 1.0
const CH_POST_MEAN_Z = 1 / 2.25 * CH_O
const CH_POST_SD_Z = sqrt(1 - 1 / 2.25)
const CH_POST_MEAN_A = 1.25 / 2.25 * CH_O
const CH_LOG_EVIDENCE = logpdf(Normal(0, sqrt(2.25)), CH_O)

# ------------------------------------------------------------------------------
# M7 branching (universal): trace structure depends on a continuous variable
#   b ~ Normal(0, 1)
#   if b > 0: x = {:c} ~ Normal(1, 1)  else: x = {:d} ~ Normal(-1, 1)
#   y ~ Normal(x, 1)   observed 0.5
#   returns b > 0
# ------------------------------------------------------------------------------
const BR_Y = 0.5
function _branching_reference()
    l_pos = 0.5 * pdf(Normal(1, sqrt(2)), BR_Y)
    l_neg = 0.5 * pdf(Normal(-1, sqrt(2)), BR_Y)
    p_pos = l_pos / (l_pos + l_neg)
    # E[x | branch, y] = (m + y)/2
    post_mean_x = p_pos * (1 + BR_Y) / 2 + (1 - p_pos) * (-1 + BR_Y) / 2
    return p_pos, post_mean_x, log(l_pos + l_neg)
end
const BR_POST_POSITIVE, BR_POST_MEAN_X, BR_LOG_EVIDENCE = _branching_reference()

# ------------------------------------------------------------------------------
# M8 geometric (universal): unbounded number of random variables
#   i = 0; while {:b => i} ~ Bernoulli(0.5) == 0; i += 1; end
#   y ~ Normal(i, 1)   observed 2.3
#   returns i
# ------------------------------------------------------------------------------
const GEO_Y = 2.3
function _geometric_reference()
    ks = 0:200
    lp = [(k + 1) * log(0.5) + logpdf(Normal(k, 1.0), GEO_Y) for k in ks]
    log_Z = logsumexp(lp)
    p = exp.(lp .- log_Z)
    return sum(p .* ks), p[1:6], log_Z # mean, P(i=0..5)
end
const GEO_POST_MEAN, GEO_POST_PROBS, GEO_LOG_EVIDENCE = _geometric_reference()

# ------------------------------------------------------------------------------
# M9 linear_regression: 2-D Gaussian posterior
#   slope ~ Normal(0, 10), intercept ~ Normal(0, 10)
#   y[i] ~ Normal(intercept + slope * x[i], 1)
# ------------------------------------------------------------------------------
const LR_XS = [1.0, 2.0, 3.0, 4.0, 5.0]
const LR_YS = [2.1, 3.9, 5.3, 7.7, 10.2]
function _linear_regression_reference()
    X = hcat(LR_XS, ones(length(LR_XS))) # [slope, intercept]
    Σ = inv(X'X + I / 100)
    μ = Σ * X' * LR_YS
    log_Z = logpdf(MvNormal(zeros(length(LR_XS)), 100 * X * X' + I), LR_YS)
    return μ, sqrt.(diag(Σ)), log_Z
end
const LR_POST_MEAN, LR_POST_SD, LR_LOG_EVIDENCE = _linear_regression_reference() # (slope, intercept)

# ------------------------------------------------------------------------------
# M10 plated_gamma_poisson: plate of non-identically distributed positive latents
#   λ[i] ~ Gamma(i + 1, 1)  for i in 1:3
#   y[i] ~ Poisson(λ[i])    for y = [2, 0, 5]
#   posterior λ[i] ~ Gamma(i + 1 + y[i], 1/2)
# ------------------------------------------------------------------------------
const PGP_YS = [2.0, 0.0, 5.0]
const PGP_POST_MEAN = [(i + 1 + PGP_YS[i]) / 2 for i in 1:3]
const PGP_POST_SD = [sqrt(i + 1 + PGP_YS[i]) / 2 for i in 1:3]

# ------------------------------------------------------------------------------
# Exact-inference specific models
# ------------------------------------------------------------------------------

# E1 two_coins: two disconnected components
#   a ~ Bernoulli(0.5); Bernoulli(a == 1 ? 0.9 : 0.1) ↦ 1
#   b ~ Bernoulli(0.5); Bernoulli(b == 1 ? 0.1 : 0.9) ↦ 1
const TC_EVIDENCE = (0.5 * 0.9 + 0.5 * 0.1) * (0.5 * 0.1 + 0.5 * 0.9) # = 0.25
const TC_POST_A1 = 0.9
const TC_POST_B1 = 0.1

# E2 orphan_observation: observation without latent parent
#   a ~ Bernoulli(0.5); Bernoulli(0.2) ↦ 1
const OO_EVIDENCE = 0.2

# E3 chain4: chain a → b → c → d, where evidence makes b = 1 impossible
#   a ~ Bernoulli(0.6)
#   b ~ Bernoulli(a == 1 ? 0.7 : 0.2)
#   c ~ Bernoulli(b == 1 ? 0.9 : 0.4)
#   d ~ Bernoulli(c == 1 ? 0.3 : 0.8)
#   Bernoulli(b == 1 ? 0.0 : 0.5) ↦ 1    (forces b = 0)
function _chain4_reference()
    pa = 0.6
    pb(a) = a == 1 ? 0.7 : 0.2
    pc(b) = b == 1 ? 0.9 : 0.4
    pd(c) = c == 1 ? 0.3 : 0.8
    po(b) = b == 1 ? 0.0 : 0.5
    marg = Dict(:a => zeros(2), :b => zeros(2), :c => zeros(2), :d => zeros(2))
    Z = 0.0
    for a in 0:1, b in 0:1, c in 0:1, d in 0:1
        p = pdf(Bernoulli(pa), a) * pdf(Bernoulli(pb(a)), b) * pdf(Bernoulli(pc(b)), c) *
            pdf(Bernoulli(pd(c)), d) * po(b)
        Z += p
        marg[:a][a+1] += p; marg[:b][b+1] += p; marg[:c][c+1] += p; marg[:d][d+1] += p
    end
    for v in values(marg)
        v ./= Z
    end
    return marg, Z
end
const C4_MARGINALS, C4_EVIDENCE = _chain4_reference()

# E4 many_observations: posterior is fine, evidence underflows in linear space
#   p ~ Categorical([0.5, 0.5]) mapped to bias 0.3 / 0.6
#   y[i] ~ Bernoulli(bias)  N = 3000 observations, 40% ones
const MO_N = 3000
const MO_YS = [i % 5 < 2 ? 1.0 : 0.0 for i in 1:MO_N]
mo_bias(z) = z == 1 ? 0.3 : 0.6
function _many_observations_reference()
    k = sum(MO_YS); n = length(MO_YS)
    lp = [log(0.5) + k * log(mo_bias(z)) + (n - k) * log(1 - mo_bias(z)) for z in 1:2]
    log_Z = logsumexp(lp)
    return exp.(lp .- log_Z), log_Z
end
const MO_POST_Z, MO_LOG_EVIDENCE = _many_observations_reference()

# E5 burglary: reference solution from examples/graph_exact_inference.jl
const BURGLARY_POST_1 = 2969983 / 992160802

end # module References
