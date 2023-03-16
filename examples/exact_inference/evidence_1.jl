using TinyPPL.Graph

function inference(show_results=false)
    model = @ppl Evidence begin
        let evidence ~ Bernoulli(0.5)
            if evidence == 1
                {:coin} ~ Bernoulli(0.5) ↦ 1.
            end
            evidence
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