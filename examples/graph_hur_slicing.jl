using TinyPPL.Graph

full_model = @pgm Full begin
    function or(x, y)
        max(x, y)
    end
    function and(x, y)
        min(x, y)
    end
    function not(x)
        1 - x
    end
    let d ~ Bernoulli(0.6),
        i ~ Bernoulli(0.7),
        g ~ Bernoulli(
            if (and(not(i), not(d)) == 1)
                0.3
            elseif (and(not(i),d) == 1)
                0.05
            elseif (and(i, not(d)) == 1)
                0.9
            else
                0.5
            end
        ),
        s ~ Bernoulli(not(i) == 1 ? 0.2 : 0.95)
        
        {:l} ~ Bernoulli(not(g) == 1 ? 0.1 : 0.4) ↦ 1

        s
    end
end;

@time f = variable_elimination(full_model, order=:Greedy)
evaluate_return_expr_over_factor(full_model, f.factor)


sliced_model = @pgm Sliced begin
    function or(x, y)
        max(x, y)
    end
    function and(x, y)
        min(x, y)
    end
    function not(x)
        1 - x
    end
    let i ~ Bernoulli(0.7),
        s ~ Bernoulli(not(i) == 1 ? 0.2 : 0.95)

        s
    end
end;

@time f = variable_elimination(sliced_model, order=:Greedy)
evaluate_return_expr_over_factor(sliced_model, f.factor)
