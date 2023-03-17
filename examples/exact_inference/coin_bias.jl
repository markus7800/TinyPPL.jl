function get_model()
    @ppl CoinBias begin
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
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 2/3, " P(1)=", 1/3)
end
