function get_model()
    @ppl Evidence begin
        let evidence ~ Bernoulli(0.5)
            if evidence == 1
                {:coin} ~ Bernoulli(0.5) ↦ 1.
            end
            evidence
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 2/3, " P(1)=", 1/3)
end