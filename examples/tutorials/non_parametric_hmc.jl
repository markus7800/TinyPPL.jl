using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using Dictionaries

# UniversalModels may instantiate an arbitary set of random variables per execution.
# There are no assumption about the program structure and this may result in an unbounded number of random variables.
# Branches are executed lazily.

@ppl function pedestrian(i_max)
    start_position ~ Uniform(0.,3.)
    position = start_position
    distance = 0.
    i = 1
    while position >= 0. && distance <= 10. && i <= i_max
        step = {:step=>i} ~ Uniform(-1.,1.)
        distance += abs(step)
        position += step
        i += 1
    end
    {:distance} ~ Normal(distance, 0.1)

    return start_position
end

args = (100,)
observations = Observations(:distance => 1.1);

Random.seed!(0)
result = non_parametric_hmc(pedestrian, args, observations, 10000, 10, 0.001);

X = Dictionary{Address,Float64}()
P = Dictionary{Address,Float64}()
Random.seed!(0)
X, grad, K = Evaluation.get_U_grad_and_extend!(pedestrian, args, observations, X, P,)

X = Dictionary{Address,Float64}()
Random.seed!(0)
X_proposed, K_current, K_proposed = Evaluation.universal_leapfrog(pedestrian, args, observations, X, 10, 0.1)


A = Dictionary{Address,Float64}()
insert!(A, :x, 1)

for i in pairs(A)
    println(i)
end

B = A .+ 1
insert!(B, :y, 2)

@which copy(A)

@macroexpand @ppl function simple(mean::Float64)
    X = {:X} ~ Normal(mean, 1.)
    {:Y} ~ Normal(X, 1.)
    return X
end