using TinyPPL.Graph

function inference(show_results=false)
    model = @ppl MurderMystery begin
        function mystery()
            let aliceDunnit ~ Bernoulli(0.3),
                withGun ~ (aliceDunnit == 1 ? Bernoulli(0.03) : Bernoulli(0.8))
                (aliceDunnit, withGun)
            end
        end
        function gunFoundAtScene(gunFound)
            let t = mystery(),
                aliceDunnit = t[1],
                withGun = t[2]

                Dirac(withGun) ↦ 1.
                aliceDunnit
            end
        end
        let posterior = gunFoundAtScene(true)
            posterior
        end
    end

    f = variable_elimination(model)
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        display(retvals)
        println("Reference: ", "P(0)=", 1-9/569, " P(1)=", 9/569)
    end
end

inference(true)

using BenchmarkTools
b = @benchmark inference()
show(Base.stdout, MIME"text/plain"(), b)

