
# https://www.reddit.com/r/math/comments/17qcx8u/the_paradox_that_broke_me/

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random


@ppl function six_in_a_row(N)
    i = 1
    nsix = 0
    while true
        x = {:x => i} ~ DiscreteUniform(1,6)
        if isodd(x)
            {:odd} ~ Dirac(true)
            break
        end
        if x == 6
            nsix += 1
        else
            nsix = 0
        end
        if nsix == N
            {:odd} ~ Dirac(false)
            break
        end
        i += 1
    end
    return i # number of dice rolls until we see `N` 6s in a row
end

@ppl function nth_six(N)
    i = 1
    nsix = 0
    while true
        x = {:x => i} ~ DiscreteUniform(1,6)
        if isodd(x)
            {:odd} ~ Dirac(true)
            break
        end
        if x == 6
            nsix += 1
        end
        if nsix == N
            {:odd} ~ Dirac(false)
            break
        end
        i += 1
    end
    return i # number of dice rolls until we see `N` 6s
end

no_odds = Observations(:odd => false)
N = 5
args = (N,)


Random.seed!(0)
retvals, lp = likelihood_weighting(six_in_a_row, args, no_odds, 10_000_000, Evaluation.retval_completion);
W = exp.(lp);
# expected number of dice rolls until we see `N` 6s in a row  GIVEN that no odds show up
retvals'W


Random.seed!(0)
retvals, lp = likelihood_weighting(nth_six, args, no_odds, 10_000_000, Evaluation.retval_completion);
W = exp.(lp);
# expected number of dice rolls until we see `N` 6s GIVEN that no odds show up
retvals'W
