function get_model()
    @pgm Evidence2 begin
        let evidence ~ Bernoulli(0.5)
            if evidence == 1
                {:coin1} ~ Bernoulli(0.5) ↦ 1.
            else
                {:coin2} ~ Bernoulli(0.5)
            end
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 1/3, " P(1)=", 2/3)
end