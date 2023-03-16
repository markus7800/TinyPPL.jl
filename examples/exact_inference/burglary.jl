using TinyPPL.Graph

function inference(show_results=false)
    model = @ppl Burglary begin
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

    f = variable_elimination(model)
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        display(retvals)
        println("Reference: ", "P(0)=", 989190819/992160802, " P(1)=", 2969983/992160802, )
    end
end

inference(true)

using BenchmarkTools
b = @benchmark inference()
show(Base.stdout, MIME"text/plain"(), b)