using TinyPPL.Graph

function inference(show_results=false)
    model = @ppl CoinBias begin
        function and(x, y)
            min(x, y)
        end
        let firstCoin ~ Bernoulli(0.5),
            secondCoin ~ Bernoulli(0.5),
            bothHeads = and(firstCoin, secondCoin)
            Dirac(bothHeads) ↦ 0.
            firstCoin
        end
    end

    f = variable_elimination(model)
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        display(retvals)
        println("Reference: ", "P(0)=", 2/3, " P(1)=", 1/3)
    end
end

inference(true)

using BenchmarkTools
b = @benchmark inference()
show(Base.stdout, MIME"text/plain"(), b)