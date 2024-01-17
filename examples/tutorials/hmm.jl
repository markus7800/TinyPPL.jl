
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

const RAINY = 1
const SUNNY = 2

const WALK = 1
const SHOP = 2
const CLEAN = 3

@ppl function simple()
    X ~ Normal(0.,1.)
    if X < 0
        return 1
    end
    Y ~ Normal(0.,1.)
    return 2
end

simple((), Forward(), Observations())

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
traces, lps = SMC(weather_guessing, args, observations, 100);
W = exp.(lps);


import Libtask
addresses_to_ix = get_address_to_ix(weather_guessing, args, observations)


p = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask = Libtask.TapedTask(weather_guessing.f, args..., p, observations)
Libtask.consume(ttask) # does work

ttask2 = Libtask.copy(ttask)
Libtask.consume(ttask)
Libtask.consume(ttask2)



@ppl static function simple(n)
    X ~ Normal(0.,1.)
    if X < n
        return 1
    end
    Y ~ Normal(0.,1.)
    return 2
end

args = (0,)
observations = Observations(:X => 1., :Y => 0.)
addresses_to_ix = get_address_to_ix(simple, args, observations)

import Libtask
p = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask = Libtask.TapedTask(simple.f, args..., p, observations)
Libtask.consume(ttask)


@ppl static function simple_sub(n)
    Z ~ Normal(0.,1.)
    @subppl simple(n)
end

args = (0,)
observations = Observations(:X => 1., :Y => 0., :Z => -1)
addresses_to_ix = get_address_to_ix(simple_sub, args, observations)

p = Evaluation.SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix);
ttask = Libtask.TapedTask(simple_sub.f, args..., p, observations)
Libtask.consume(ttask)

ttask2 = copy(ttask)
Libtask.consume(ttask2)
