
using TinyPPL.Distributions
using TinyPPL.Graph
import Random


# Dice Holtzen Section 3.2.2

obsprog = @pgm ObsProg begin
    function or(x, y)
        max(x, y)
    end
    let x ~ Bernoulli(0.6),
        y ~ Bernoulli(0.3)
        Dirac(or(x,y)) ↦ 1
        x
    end
end

@time traces, lps = likelihood_weighting(obsprog, 1_000_000);
W = exp.(lps);
traces.retvals'W
p = [0.6, 0.12] / 0.72

# P(X = 1 | X || Y = 1)  = P( X = 1,  X || Y = 1) / P(X || Y = 1) = P(X = 1) / (1 - P(X=0, Y=0))
0.6 / (1 - 0.4*0.7)

# f and g return 1 with probability 1
f_model = @pgm f_model begin
    function or(x, y)
        max(x, y)
    end
    function f(x)
        let flip ~ Bernoulli(0.5),
            y = or(x, flip)

            Dirac(y) ↦ 1
            y
        end
    end
    let x ~ Bernoulli(0.1),
        obs = f(x)
        x
    end
end

@time traces, lps = likelihood_weighting(f_model, 1_000_000);
W = exp.(lps);
traces.retvals'W
0.1/0.55

g_model = @pgm g_model begin
    function g(x)
        1
    end
    let x ~ Bernoulli(0.1),
        obs = g(x)
        x
    end
end


@time traces, lps = likelihood_weighting(g_model, 1_000_000);
W = exp.(lps);
traces.retvals'W
0.1