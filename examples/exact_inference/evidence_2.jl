using TinyPPL.Graph

function inference(show_results=false)
    model = @ppl Evidence2 begin
        let evidence ~ Bernoulli(0.5)
            if evidence == 1
                {:coin1} ~ Bernoulli(0.5) ↦ 1.
            else
                {:coin2} ~ Bernoulli(0.5)
            end
        end
    end

    f = variable_elimination(model)
    retvals = evaluate_return_expr_over_factor(model, f)

    if show_results
        display(retvals)
        println("Reference: ", "P(0)=", 1/3, " P(1)=", 2/3)
    end
end

inference(true)

using BenchmarkTools
b = @benchmark inference()
show(Base.stdout, MIME"text/plain"(), b)
