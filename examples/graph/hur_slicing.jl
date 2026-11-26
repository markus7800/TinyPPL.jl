# Slicing probabilistic programs (Hur, Nori, Rajamani, Samuel: Slicing Probabilistic Programs, PLDI 2014).
#
# Slicing removes the parts of a program that cannot influence the result, which makes inference cheaper.
# In the student network below, we ask whether the student gets a good SAT score s, which depends only on intelligence i.
# Without observations, difficulty d, grade g and letter l can be sliced away.
# But observing l = 1 makes i relevant through the grade g, which depends on both i and d.
# So a correct slice must keep everything that the observation depends on, not just what s depends on.
#
# Run: julia --project=examples examples/graph/hur_slicing.jl

using TinyPPL.Distributions
using TinyPPL.Graph

student = @pgm student begin
    function and(x, y)
        min(x, y)
    end
    function not(x)
        1 - x
    end
    let d ~ Bernoulli(0.6),
        i ~ Bernoulli(0.7),
        g ~ Bernoulli(
            if and(not(i), not(d)) == 1
                0.3
            elseif and(not(i), d) == 1
                0.05
            elseif and(i, not(d)) == 1
                0.9
            else
                0.5
            end
        ),
        s ~ Bernoulli(not(i) == 1 ? 0.2 : 0.95),
        l ~ Bernoulli(not(g) == 1 ? 0.1 : 0.4)

        s
    end
end

student_observed = @pgm student_observed begin
    function and(x, y)
        min(x, y)
    end
    function not(x)
        1 - x
    end
    let d ~ Bernoulli(0.6),
        i ~ Bernoulli(0.7),
        g ~ Bernoulli(
            if and(not(i), not(d)) == 1
                0.3
            elseif and(not(i), d) == 1
                0.05
            elseif and(i, not(d)) == 1
                0.9
            else
                0.5
            end
        ),
        s ~ Bernoulli(not(i) == 1 ? 0.2 : 0.95)

        {:l} ~ Bernoulli(not(g) == 1 ? 0.1 : 0.4) ↦ 1
        s
    end
end

# The slice keeps only i and s.
sliced = @pgm sliced begin
    function not(x)
        1 - x
    end
    let i ~ Bernoulli(0.7),
        s ~ Bernoulli(not(i) == 1 ? 0.2 : 0.95)

        s
    end
end

# Variable elimination gives a factor over the variables of the return expression,
# evaluate_return_expr_over_factor turns it into (value, probability) pairs.
P_s(model) = Dict(evaluate_return_expr_over_factor(model, variable_elimination(model, order=:Greedy).factor))[1.]

println("P(s = 1):         full model ", round(P_s(student), digits=4), ", slice ", round(P_s(sliced), digits=4))
println("P(s = 1 | l = 1): full model ", round(P_s(student_observed), digits=4), ", slice ", round(P_s(sliced), digits=4), " (the slice is wrong)")
