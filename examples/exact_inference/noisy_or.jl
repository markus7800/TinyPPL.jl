function get_model()
    @ppl NoisyOr begin
        function or(x, y)
            max(x, y)
        end
        let n0 ~ Bernoulli(0.5),
            n4 ~ Bernoulli(0.5),
            n1 ~ Bernoulli(n0 == 1 ? 4/5 : 1/10),
            n21 ~ Bernoulli(n0 == 1 ? 4/5 : 1/10),
            n22 ~ Bernoulli(n4 == 1 ? 4/5 : 1/10),
            n33 ~ Bernoulli(n4 == 1 ? 4/5 : 1/10),
            n2 = or(n21, n22),
            n31 ~ Bernoulli(n1 == 1 ? 4/5 : 1/10),
            n32 ~ Bernoulli(n2 == 1 ? 4/5 : 1/10),
            n3 = or(or(n31, n32), n33)

            n3
        end
    end
end

function print_reference_solution()
    println("Reference: ",  "P(0)=", 29693/160000, " P(1)=", 130307/160000)
end