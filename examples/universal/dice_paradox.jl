# A dice paradox (https://www.reddit.com/r/math/comments/17qcx8u/the_paradox_that_broke_me/, often attributed to Elchanan Mossel).
#
# Roll a fair die until you see N sixes in a row. What is the expected number of rolls, given that all rolls were even?
# The intuitive answer is to roll a three-sided die {2, 4, 6} until N sixes in a row, which gives 12 rolls for N = 2.
# But the condition "all rolls were even" favours short sequences, because every additional roll risks an odd number.
# The correct answer is 30/11 ≈ 2.73 for N = 2, which is even less than for N sixes that are not necessarily in a row (3).
# Without the condition, it takes 42 rolls on average to see two sixes in a row (and 12 for two sixes).
#
# To condition on all rolls being even, we stop rolling at the first odd number
# and observe that the sequence stopped because of the sixes.
#
# Run: julia --project=examples examples/universal/dice_paradox.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import LinearAlgebra: I

@ppl function roll_until_sixes(N::Int, in_a_row::Bool, condition_on_even::Bool)
    i = 1
    n_six = 0
    while true
        x = {:x => i} ~ DiscreteUniform(1, 6)
        if condition_on_even && isodd(x)
            {:odd} ~ Dirac(true)
            break
        end
        if x == 6
            n_six += 1
        elseif in_a_row
            n_six = 0
        end
        if n_six == N
            if condition_on_even
                {:odd} ~ Dirac(false)
            end
            break
        end
        i += 1
    end
    return i # number of rolls
end

# Exact E[number of rolls | no odd roll] by first-step analysis over the states s = 0, ..., N-1 (number of sixes so far).
# From state s, a 6 (probability 1/6) moves to s+1, and any other allowed number (2 or 4 with probability 2/6 when
# conditioning on even rolls, otherwise 1 to 5 with probability 5/6) moves to 0 (in a row) or stays at s.
# With A the transition matrix between these states and b the probability of reaching N in one roll,
# the success probabilities P = A P + b and M = E[rolls * 1{success}] = A (P + M) + b = P + A M.
# Without conditioning, P = 1 and M is the expected number of rolls.
# We compute with exact rational numbers: in floating point, the solution loses all precision for large N,
# e.g. 100 sixes in a row take ~6^100 rolls on average, and I - A is then numerically singular.
function exact_expected_rolls(N::Int, in_a_row::Bool, condition_on_even::Bool)
    p_other = condition_on_even ? 2//6 : 5//6
    A = zeros(Rational{BigInt}, N, N) # state s has index s+1
    b = zeros(Rational{BigInt}, N)
    for s in 0:N-1
        A[s+1, (in_a_row ? 0 : s)+1] += p_other # no six, stay on count if not in a row or move to 0 if in a row
        s + 1 == N ? (b[s+1] += 1//6) : (A[s+1, s+2] += 1//6) # six, increment count or reach success state
    end
    # P[s] is the success probability when starting with count s
    # P[s] = p_other * (P[s] or P[0]) + 1/6 * P[s+1]
    # P[N-1] = p_other * (P[N-1] or P[0]) + 1/6
    # thus, P = A P + b
    P = (I - A) \ b
    # Let T_s be the number of rolls until we stop, when starting with count s, and
    # M[s] = E[T_s * 1{success}] the expected number of rolls where failed sequences count as 0.
    # The first roll either stops the sequence or continues it with a new count s'
    # (s' = s+1 after a six, s' = 0 or s after another allowed number).
    # Stopping contributes 0 (odd roll, failure) or 1 (six with count N-1, success after T_s = 1 roll).
    # Continuing, the remaining rolls behave exactly like a sequence started with count s', since the rolls are
    # independent. Thus T_s = 1 + T_s' and the sequence succeeds iff the continued sequence succeeds, so this contributes
    # E[(1 + T_s') * 1{success}] = P[s'] + M[s'].
    # M[s] = p_other * (P[s] + M[s] or P[0] + M[0]) + 1/6 * (P[s+1] + M[s+1])
    # M[N-1] = p_other * (P[N-1] + M[N-1] or P[0] + M[0]) + 1/6
    # thus, M = A (P + M) + b = (A P + b) + A M = P + A M, i.e. (I - A) M = P
    M = (I - A) \ P
    # E[T_0 | success] = E[T_0 * 1{success}] / P(success) = M[0] / P[0], at index 1
    return Float64(M[1] / P[1])
end

no_odds = Observations(:odd => false)
for N in [2, 5], in_a_row in [true, false]
    sixes = string(N, " sixes", in_a_row ? " in a row" : "")

    # Without observations, likelihood weighting samples from the prior and all weights are equal.
    # Five sixes in a row take 9330 rolls on average, so we use fewer samples.
    Random.seed!(0)
    n_rolls, _ = likelihood_weighting(roll_until_sixes, (N, in_a_row, false), Observations(), 10_000; completion=Evaluation.retval_completion)
    println("E[rolls until ", sixes, "] ≈ ", round(mean(n_rolls), digits=3),
        " (exact: ", round(exact_expected_rolls(N, in_a_row, false), digits=3), ")")

    Random.seed!(0)
    n_rolls, logw = likelihood_weighting(roll_until_sixes, (N, in_a_row, true), no_odds, 10_000_000; completion=Evaluation.retval_completion)
    W = exp.(logw)
    println("E[rolls until ", sixes, " | all even] ≈ ", round(n_rolls'W, digits=3),
        " (exact: ", round(exact_expected_rolls(N, in_a_row, true), digits=3), ", effective sample size: ", round(Int, 1 / sum(W .^ 2)), ")")

    println()
end


N = 100
for in_a_row in [true, false]
    sixes = string(N, " sixes", in_a_row ? " in a row" : "")
    println("E[rolls until ", sixes, "] = ", round(exact_expected_rolls(N, in_a_row, false), digits=3), " (exact)")
    println("E[rolls until ", sixes, " | all even] = ", round(exact_expected_rolls(N, in_a_row, true), digits=3), " (exact)")
end