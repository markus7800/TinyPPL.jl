function get_model()
    @pgm Burglary begin
        function or(x, y)
            max(x, y)
        end
        function and(x, y)
            min(x, y)
        end
        let earthquake ~ Bernoulli(0.0001),
            burglary ~ Bernoulli(0.001),
            alarm = or(earthquake, burglary),
            phoneWorking ~ (earthquake == 1 ? Bernoulli(0.7) : Bernoulli(0.99)),
            maryWakes ~ (
                if alarm == 1 
                    if earthquake == 1
                        Bernoulli(0.8)
                    else
                        Bernoulli(0.6)
                    end
                else
                    Bernoulli(0.2)
                end
            ),
            called = and(maryWakes, phoneWorking)

            Dirac(called) ↦ 1.
            burglary
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 989190819/992160802, " P(1)=", 2969983/992160802)
end