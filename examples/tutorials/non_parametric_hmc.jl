using TinyPPL.Distributions
using TinyPPL.Evaluation
using TinyPPL.Logjoint
import Random
using Dictionaries
using Plots

function get_vals(result, addr)
    [get(r, addr, missing) for r in result]
end

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


@ppl function test()
    X ~ Normal(0.,1.)
    if X < 0.
        Y ~ Normal(0.,1)
    else
    end
end
args = ()
observations = Observations()

xs = -3:0.1:3
ys = -3:0.1:3
U = [Evaluation.get_U(test, args, observations, Dictionary{Address,Float64}([:X,:Y], [x, y]), nothing)[1] for y in ys, x in xs];
heatmap(xs, ys, U)

traces, lps = likelihood_weighting(test, args, observations, 100_000);
histogram(traces[:X], weights = exp.(lps))
histogram(traces[:Y], weights = exp.(lps)[.!ismissing.(traces[:Y])])

traces, lps = rwmh(test, args, observations, 100_000);
histogram(traces[:X])
histogram(traces[:Y])
mean(traces[:X] .< 0)

Random.seed!(0)
result = non_parametric_hmc(test, args, observations, 10000, 10, 0.1);
histogram(get_vals(result, :X))
histogram(get_vals(result, :Y))

X_current = Dictionary{Address,Float64}([:X], [0.])
U_current = Evaluation.get_U(test, args, observations, X_current, nothing)

Random.seed!(0)
X_proposed, K_current, K_proposed = Evaluation.universal_leapfrog(test, args, observations, X_current, 1, 0.1)
U_proposed = Evaluation.get_U(test, args, observations, X_proposed, nothing)

K_current + U_current - K_proposed - U_proposed

@ppl static function static_test()
    X ~ Normal(0.,1.)
    if X <= 0.
        Y ~ Normal(0.,1)
    end
end
args = ()
observations = Observations()

logjoint, addresses_to_ix = Evaluation.make_unconstrained_logjoint(static_test, args, observations)
K = length(addresses_to_ix)
Random.seed!(0)
samples = hmc_logjoint(logjoint, K, 1000000, 1, 0.01, ad_backend=:forwarddiff);
traces = StaticTraces(addresses_to_ix, samples, Any[])
histogram(traces[:X])
histogram(traces[:Y])

grad_U = Logjoint.get_grad_U_fwd_diff(logjoint)

X_current = [0.1,1.5]
P_current = [-0.1,0.]
X_proposed, P_proposed = Logjoint.leapfrog(grad_U, X_current, P_current, 10, 0.1)

exp((-logjoint(X_current) + sum(P_current.^2))-(-logjoint(X_proposed) + sum(P_proposed.^2)))


P = Dictionary{Address,Float64}()
Random.seed!(0)
X, grad, K = Evaluation.get_U_grad_and_extend!(test, args, observations, X, P,)

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

using Turing

@model function discontinuous()
    X ~ Normal(0.,1.)
    if X <= 0.
        Y ~ Normal(0.,1)
    end
end

Turing.Random.seed!(1)
res = Turing.sample(discontinuous(), HMC(0.1, 10), 100000);
histogram(res[:X])
histogram(res[:Y])