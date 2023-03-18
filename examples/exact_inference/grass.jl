function get_model()
    @ppl Grass begin
        function or(x, y)
            max(x, y)
        end
        function and(x, y)
            min(x, y)
        end
        let cloudy ~ Bernoulli(0.5),
            rain ~ Bernoulli(cloudy == 1 ? 0.8 : 0.2),
            sprinkler ~ Bernoulli(cloudy == 1 ? 0.1 : 0.5),
            #temp1 ~ Bernoulli(0.7),
            #wetRoof = and(temp1, rain),
            temp2 ~ Bernoulli(0.9),
            temp3 ~ Bernoulli(0.9),
            wetGrass = or(and(temp2, rain), and(temp3, sprinkler))

            Dirac(wetGrass) ↦ 1.
            rain
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 1-509/719, " P(1)=", 509/719)
end