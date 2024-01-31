
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

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




begin
    # xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
    xs = [-1., -0.5, 0.0, 0.5, 1.0];
    ys = [-3.2, -1.8, -0.5, -0.2, 1.5];


    slope_prior_mean = 0
    slope_prior_sigma = 3
    intercept_prior_mean = 0
    intercept_prior_sigma = 3

    σ = 2.0
    m0 = [0., 0.]
    S0 = [intercept_prior_sigma^2 0.; 0. slope_prior_sigma^2]
    Phi = hcat(fill(1., length(xs)), xs)
    map_Σ = inv(inv(S0) + Phi'Phi / σ^2) 
    map_mu = map_Σ*(inv(S0) * m0 + Phi'ys / σ^2)
    map_sigma = [sqrt(map_Σ[1,1]), sqrt(map_Σ[2,2])]
    println("map_mu")
    display(map_mu)
    println("map_Σ")
    display(map_Σ)
end

function f(slope, intercept, x)
    intercept + slope * x
end

@ppl static function LinRegStatic(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end
end

args = (xs,)
observations = Observations((:y, i) => y for (i, y) in enumerate(ys));


Random.seed!(0)
traces, lps = likelihood_weighting(LinRegStatic, args, observations, 10_000);
W = exp.(lps);


Random.seed!(0)
@time traces, lps = smc(LinRegStatic, args, observations, 10_000);
W = exp.(lps);

println(W'traces[:intercept], " vs true ", map_mu[1])
println(W'traces[:slope], " vs true ", map_mu[2])

X = Real[W'traces[:intercept], W'traces[:slope]]
Random.seed!(0)
traces, lps = conditional_smc(LinRegStatic, args, observations, 1, X);
traces.data[:,1] == X

Random.seed!(0)
traces, lps = conditional_smc(LinRegStatic, args, observations, 100, X);
W = exp.(lps);
traces.data[:,1] == X


Random.seed!(0)
traces = particle_gibbs(LinRegStatic, args, observations, 100, 1000)
println(mean(traces[:intercept]), " vs true ", map_mu[1])
println(mean(traces[:slope]), " vs true ", map_mu[2])


Random.seed!(0)
traces = particle_IMH(LinRegStatic, args, observations, 100, 1000)


using Plots
scatter(traces[:intercept], traces[:slope])
import Distributions
posterior = Distributions.MvNormal(map_mu, map_Σ)
posterior_sample = rand(posterior, 1000)
scatter(posterior_sample[1,:], posterior_sample[2,:],)


# Random.TaskLocalRNG()

# Any[(:y, 1)]
# Neff=4681.61187053709
# Any[(:y, 2)]
# Neff=9064.870236270977
# Any[(:y, 3)]
# Neff=8588.965923415002
# Any[(:y, 4)]
# Neff=8423.38368401471
# Any[(:y, 5)]
# Neff=8232.574091256161
# Any[:__BREAK]
#   5.220231 seconds (8.05 M allocations: 430.802 MiB, 4.89% gc time)


# -0.7402298247713963 vs true -0.7714285714285714

# 1.8221096950814604 vs true 1.8679245283018866


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

T = 3
args = (T,)
a = 0.9
σ_v = 0.32
σ_e = 1.
update_freq(traces, n_samples) = [mean(traces[:x=>t, i] != traces[:x=>t, i+1] for i in 1:n_samples-1) for t in 1:T]


Random.seed!(0)
traces = sample_from_prior(LGSS, args, 10^7)

import Distributions: var, cov
samples = convert(Matrix{Float64}, traces.data)
mean(samples, dims=2)
var(samples, dims=2)
C = cov(transpose(samples))

observations = Observations((:y => t) => traces[:y => t, 1] for t in 1:T)
ground_truth = Dict((:x => t) => traces[:x => t, 1] for t in 1:T)

for t in 1:T
    println(observations[:y => t], ",")
end

for t in 1:T
    println(ground_truth[:x => t], ",")
end

using Plots
plot([observations[:y => t] for t in 1:T]);
plot!([ground_truth[:x => t] for t in 1:T])

n_particles = 5
n_samples = 1000
Random.seed!(0)
@time traces = particle_gibbs(LGSS, args, observations, n_particles, n_samples; ancestral_sampling=false);

plot(1:T, fill((n_particles-1) / n_particles, T));
plot!(update_freq(traces, n_samples), ylim=(0,1))


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

    x_last = X[addresses_to_ix[:x => (t-1)]]
    y_t = dist.observations[:y => t]

    mu_cond = a*x_last + σ_v^2 / (σ_e^2+σ_v^2) * (y_t - a*x_last)
    sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))

    return Normal(mu_cond, sigma_cond)
end

addr2proposal = Addr2Proposal((:x => t) => LGSSProposal(a, σ_v, σ_e, observations) for t in 2:T);
addr2proposal[:x => 1] = StaticProposal(Normal(0, σ_v))

n_particles = 5
n_samples = 1000
Random.seed!(0)
@time traces = particle_gibbs(LGSS, args, observations, n_particles, n_samples; ancestral_sampling=false, addr2proposal=addr2proposal);

plot(1:T, fill((n_particles-1) / n_particles, T));
plot!(update_freq(traces, n_samples), ylim=(0,1))


import Libtask
addresses_to_ix = get_address_to_ix(weather_guessing, args, observations)


p = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask = Libtask.TapedTask(weather_guessing.f, args..., p, observations)
Libtask.consume(ttask) # does work

ttask2 = Libtask.copy(ttask)
Libtask.consume(ttask)
Libtask.consume(ttask2)

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

@ppl static function simple(a, b)
    X ~ Normal(0.,1.)
    if X < (a+b)
        return 1
    end
    Y ~ Normal(0.,1.)
    return 2
end

args = (0,0)
observations = Observations(:X => 1., :Y => 0.)
addresses_to_ix = get_address_to_ix(simple, args, observations)

import Libtask
p = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask = Libtask.TapedTask(simple.f, args..., p, observations)
Libtask.consume(ttask)


@ppl static function simple_sub(n)
    Z ~ Normal(0.,1.)
    @subppl simple(n, 0)
end

args = (0,)
observations = Observations(:X => 1., :Y => 0., :Z => -1)
addresses_to_ix = get_address_to_ix(simple_sub, args, observations)

p = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask = Libtask.TapedTask(simple_sub.f, args..., p, observations)
_,_,_p = Libtask.consume(ttask)
_p === p

p2 = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask2 = copy(ttask)
Evaluation.update_particle!(ttask2, p2)

_,_,_p = Libtask.consume(ttask2)
_p === p2
ix = ttask2.tf.arg_binding_slots[3]
ttask2.tf.binding_values[ix] === p2

sub_tf = collect(values(ttask2.tf.subtapes))[1]
ix2 = sub_tf.arg_binding_slots[4]
sub_tf.binding_values[ix2] === p2

