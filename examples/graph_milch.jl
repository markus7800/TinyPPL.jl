

# Milch et. al. Approximate Inference for Infinite Contingent Bayesian Networks

using TinyPPL.Distributions
using TinyPPL.Graph
using TinyPPL: logsumexp
import Random

model = @pgm urn begin
    let N = 3, K = 10,
        balls = [{:ball => i} ~ Bernoulli(0.5) for i in 1:N],
        ball_ixs = [{:drawn_ball => k} ~ DiscreteUniform(1,N) for k in 1:K],
        n_black = sum([balls[Int(ball_ixs[k])] for k in 1:K])
        {:n_black} ~ Dirac(n_black) ↦ 1
        n_black
    end
end

variable_nodes, factor_nodes = get_factor_graph(model)

f = factor_nodes[end]
size(f.table)
prod(size(f.table))
sum(f.table .== -Inf)
sum(isfinite.(f.table))

mean(isfinite.(f.table))

factor_size(N, K) = 2^N * N^K
factor_size(3,10)
factor_size(6,10)

obs = repeat([0,1], 5)
model = @pgm urn begin
    let N = 6, K = 10,
        balls = [{:ball => i} ~ Bernoulli(0.5) for i in 1:N],
        ball_ixs = [{:drawn_ball => k} ~ DiscreteUniform(1,N) for k in 1:K]
        # [Dirac(balls[Int(ball_ixs[k])]) ↦ $(Main.obs)[k] for k in 1:K] # unbiased
        [Bernoulli(balls[Int(ball_ixs[k])] == 1. ? 0.8 : 0.2) ↦ $(Main.obs)[k] for k in 1:K] # biased
        0
    end
end

function logaddexp(x1, x2)
    xstar = max(x1, x2)
    return xstar + log(exp(x1 - xstar) + exp(x2 - xstar))
end 

function true_log_joint(N::Int, biased::Bool, obs::Vector{Int})
    # n_black ~ Binomial(N,0.5)
    # balls[Int(ball_ixs[k]) ~ Bernoulli(n_black/N)
    # llik = 5 * pdf(Bernoulli(n_black/N), 1) + 5*pdf(Bernoulli(n_black/N), 0)
    if !biased && N == 1
        return -Inf
    end
    n_obs0 = sum(obs .== 0)
    n_obs1 = sum(obs .== 1)
    lp = -Inf
    for n_black in 0:N
        lp_n_black = logpdf(Distributions.Binomial(N, 0.5), n_black)
        if biased
            d = Distributions.Bernoulli(n_black/N)
            lp_obs0 = logaddexp(logpdf(Distributions.Bernoulli(0.8), 0) + logpdf(d, 1), logpdf(Distributions.Bernoulli(0.2), 0) + logpdf(d, 0))
            lp_obs1 = logaddexp(logpdf(Distributions.Bernoulli(0.8), 1) + logpdf(d, 1), logpdf(Distributions.Bernoulli(0.2), 1) + logpdf(d, 0))
            lp_n_black += n_obs0 * lp_obs0 + n_obs1 * lp_obs1
        else
            d = Distributions.Bernoulli(n_black/N)
            lp_n_black += n_obs0 * logpdf(d, 1) + n_obs1 * logpdf(d, 0)
        end
        lp = log(exp(lp) + exp(lp_n_black))
    end
    lp += logpdf(Distributions.Poisson(6), N)
    return lp
end
lpost = [true_log_joint(N, true) for N in 1:1000];
exp.(lpost .- logsumexp(lpost))[1:20]

variable_nodes, factor_nodes = get_factor_graph(model);
greedy_variable_elimination(variable_nodes, Int[]).evidence

llik_biased = [
    -9.16290731874155,
    -7.52262737473222,
    -7.380304224799481,
    -7.287521451720479,
    -7.227047233966903,
    -7.1844611773009746,
    -7.152770354939277,
    -7.128226374003257,
    -7.108635195225264,
    -7.092623490061302,
    -7.079285815785726,
    -7.068000041762838,
    -7.05832408345374,
    -7.049934897967121,
    -7.0425907935903975
]

llik_unbiased = [
    -Inf,
    -7.624618986159398,
    -7.808069056333151,
    -7.637395572738379,
    -7.541818789087474,
    -7.471819186788574,
    -7.417060683095864,
    -7.37279373086686,
    -7.336163626942312,
    -7.305296303234316,
    -7.278898962391384,
    -7.256046569354112,
    -7.2360575453488405,
    -7.2184173709226815,
    -7.2027296221444095
]
import Distributions
ljoint = llik_unbiased + logpdf(Distributions.Poisson(6), 1:15)
lpost = ljoint .- logsumexp(ljoint)
exp.(lpost)


f = factor_nodes[end]
size(f.table)
prod(size(f.table))
sum(f.table .== -Inf)
sum(isfinite.(f.table))

factor_size(N, K) = 2^N * N
factor_size(3,10)
factor_size(15,10)

num_factors(N, K) = N + K + K
num_factors(3,10)
num_factors(6,10)
