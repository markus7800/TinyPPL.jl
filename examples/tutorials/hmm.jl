
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

observations = Observations((:activity => i) => traces[:activity => i][1] for i in 1:N)
ground_truth = Dict((:weather => i) => traces[:weather => i][1] for i in 0:N)


Random.seed!(0)
traces, lps = likelihood_weighting(weather_guessing, args, observations, 1_000_000);
W = exp.(lps);

for i in 0:N
    println(W'traces[:weather => i], " vs true ", ground_truth[:weather => i])
end


Random.seed!(0)
traces, lps = SMC(weather_guessing, args, observations, 1000);
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
traces, lps = SMC(LinRegStatic, args, observations, 10_000);
W = exp.(lps);

println(W'traces[:intercept], " vs true ", map_mu[1])
println(W'traces[:slope], " vs true ", map_mu[2])

length(unique(zip(traces[:intercept], traces[:slope])))




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

