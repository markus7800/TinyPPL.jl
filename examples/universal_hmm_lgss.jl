
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import Distributions: mean, std
import LinearAlgebra: diag

const RAINY = 1
const SUNNY = 2

const WALK = 1
const SHOP = 2
const CLEAN = 3

@ppl static function weather_guessing(N)
    weather = {:weather => 0} ~ Categorical([0.6, 0.4]) # 1 ... Rainy, 2 ... Sunny
    for i in 1:N
        # transition
        if weather == RAINY
            weather = {:weather => i} ~ Categorical([0.7, 0.3])
        else
            weather = {:weather => i} ~ Categorical([0.4, 0.6])
        end
        # emission
        if weather == RAINY
            a = {:activity => i} ~ Categorical([0.1, 0.4, 0.5]) # 1 ... Walk, 2 ... Shop, 3 ... Clean
        else
            a = {:activity => i} ~ Categorical([0.6, 0.3, 0.1])
        end
    end
end
N = 5
args = (N,)

Random.seed!(0)
traces = sample_from_prior(weather_guessing, args, Observations(), 1)

observations = Observations((:activity => i) => traces[:activity => i, 1] for i in 1:N)
ground_truth = Dict((:weather => i) => traces[:weather => i][1] for i in 0:N)


Random.seed!(0)
traces, lps = likelihood_weighting(weather_guessing, args, observations, 1_000_000);
W = exp.(lps);

for i in 0:N
    println(W'traces[:weather => i], " vs true ", ground_truth[:weather => i])
end


Random.seed!(0)
traces, lps = smc(weather_guessing, args, observations, 1000);
W = exp.(lps);

for i in 0:N
    println(W'traces[:weather => i], " vs true ", ground_truth[:weather => i])
end



include("lgss/lgss_data.jl")
include("lgss/posterior.jl")

@ppl static function LGSS(T)
    a = 0.9
    σ_v = 0.32
    σ_e = 1.

    x = 0
    for t in 1:T
        x = {:x => t} ~ Normal(a * x, σ_v)
        {:y => t} ~ Normal(x, σ_e)
    end
end

T = 400
args = (T,)
a = 0.9
σ_v = 0.32
σ_e = 1.
update_freq(traces, n_samples) = [mean(traces[:x=>t, i] != traces[:x=>t, i+1] for i in 1:n_samples-1) for t in 1:T]

# to verify get_joint_normal
# import Distributions: var, cov
# traces = sample_from_prior(LGSS, args, 10^7)
# samples = convert(Matrix{Float64}, traces.data)
# mean(samples, dims=2)
# var(samples, dims=2)
# C = cov(transpose(samples))

# used to generate data
# Random.seed!(0)
# traces = sample_from_prior(LGSS, args, 1)

# observations = Observations((:y => t) => traces[:y => t, 1] for t in 1:T)
# ground_truth = Dict((:x => t) => traces[:x => t, 1] for t in 1:T)

# for t in 1:T
#     println(observations[:y => t], ",")
# end

# for t in 1:T
#     println(ground_truth[:x => t], ",")
# end


observations = Observations((:y => t) => y[t] for t in 1:T)
ground_truth = Dict((:x => t) => x_gt[t] for t in 1:T)


using Plots
plot([observations[:y => t] for t in 1:T]);
plot!([ground_truth[:x => t] for t in 1:T])

n_particles = 5
n_samples = 1000
Random.seed!(0)
@time traces = particle_gibbs(LGSS, args, observations, n_particles, n_samples; ancestral_sampling=true);

plot(1:T, fill((n_particles-1) / n_particles, T), lc=1, legend=false);
plot!(update_freq(traces, n_samples), ylim=(0,1), lc=1)

# visualise true posterior
mu_post, Σ_post = get_true_posterior(T, y)

inferred_mean = mean(traces.data, dims=2)
inferred_std = std(traces.data, dims=2)

# even 5 particels approximate posterior well
plot(inferred_mean, ribbon=2*inferred_std, label="estimated posterior",fillalpha=0.3);
plot!(mu_post, ribbon=2*sqrt.(diag(Σ_post)), label="true posterior", fillalpha=0.3)
plot!(x_gt, label="ground truth")
# plot!(y, label="observed")

struct LGSSProposal <: Distributions.ProposalDistribution
    a::Float64
    σ_v::Float64
    σ_e::Float64
    observations::Observations
end

function Distributions.proposal_dist(dist::LGSSProposal, x_current::Tuple{Pair,StaticTrace,Evaluation.Addr2Ix})
    addr, X, addresses_to_ix  = x_current
    t::Int = addr[2]
    a = dist.a
    σ_v = dist.σ_v
    σ_e = dist.σ_e

    x_last = t == 1 ? 0. : X[addresses_to_ix[:x => (t-1)]]
    y_t = dist.observations[:y => t]

    mu_cond = a*x_last + σ_v^2 / (σ_e^2+σ_v^2) * (y_t - a*x_last)
    sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))

    return Normal(mu_cond, sigma_cond)
end

addr2proposal = Addr2Proposal((:x => t) => LGSSProposal(a, σ_v, σ_e, observations) for t in 1:T);

n_particles = 5
n_samples = 1000
Random.seed!(0)
@time traces = particle_gibbs(LGSS, args, observations, n_particles, n_samples; ancestral_sampling=false, addr2proposal=addr2proposal);

plot(1:T, fill((n_particles-1) / n_particles, T), lc=1, legend=false);
plot!(update_freq(traces, n_samples), ylim=(0,1), lc=1)
