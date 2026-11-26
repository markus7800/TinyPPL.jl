# A hidden Markov model: guessing the weather from someone's daily activities.
#
# The weather (rainy or sunny) follows a Markov chain, and each day the person walks, shops or cleans,
# with probabilities depending on the weather. We observe the activities of N = 5 days and infer the weather.
# With only 2^6 weather sequences, the exact posterior can be computed by enumeration.
#
# Run: julia --project=examples examples/static/hmm.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

const RAINY, SUNNY = 1, 2
const WALK, SHOP, CLEAN = 1, 2, 3

const initial = [0.6, 0.4]                      # P(weather_0)
const transition = [0.7 0.3; 0.4 0.6]           # transition[w, w'] = P(weather_i = w' | weather_{i-1} = w)
const emission = [0.1 0.4 0.5; 0.6 0.3 0.1]     # emission[w, a] = P(activity_i = a | weather_i = w)

@ppl static function weather_guessing(N)
    weather = {:weather => 0} ~ Categorical(initial)
    for i in 1:N
        weather = {:weather => i} ~ Categorical(transition[Int(weather), :])
        {:activity => i} ~ Categorical(emission[Int(weather), :])
    end
end

N = 5
args = (N,)

# Simulate the weather and activities from the model.
Random.seed!(0)
trace = sample_from_prior(weather_guessing, args, Observations(), 1)
activities = [Int(trace[:activity => i, 1]) for i in 1:N]
true_weather = [Int(trace[:weather => i, 1]) for i in 0:N]
observations = Observations((:activity => i) => activities[i] for i in 1:N)
println("activities: ", [("walk", "shop", "clean")[a] for a in activities])
println("weather:    ", [("rainy", "sunny")[w] for w in true_weather])

# Exact posterior P(weather_i = rainy | activities) by summing the joint probability over all weather sequences.
function exact_rainy_probabilities()
    p_rainy = zeros(N + 1)
    p_total = 0.
    for weather in Iterators.product(fill(1:2, N + 1)...) # weather[i+1] = weather_i
        p = initial[weather[1]] * prod(transition[weather[i], weather[i+1]] * emission[weather[i+1], activities[i]] for i in 1:N)
        p_rainy .+= p .* (collect(weather) .== RAINY)
        p_total += p
    end
    return p_rainy ./ p_total
end

println("\nP(weather_i = rainy | activities) for i = 0:", N)
println("  exact                ", round.(exact_rainy_probabilities(), digits=3))

Random.seed!(0)
traces, logw = likelihood_weighting(weather_guessing, args, observations, 1_000_000)
W = exp.(logw)
println("  likelihood weighting ", [round(sum(W[traces[:weather => i] .== RAINY]), digits=3) for i in 0:N])

# SMC resamples after each observed activity.
Random.seed!(0)
traces, logw = smc(weather_guessing, args, observations, 10_000)
W = exp.(logw)
println("  SMC                  ", [round(sum(W[traces[:weather => i] .== RAINY]), digits=3) for i in 0:N])
