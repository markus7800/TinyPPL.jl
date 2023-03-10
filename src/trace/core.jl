
import MacroTools
import Distributions: logpdf

struct RV
    value::Real
    logprob::Float64
end

const Trace = Dict{Any, RV}

export RV, Trace

macro subppl(expr) return expr end # just a place holder

function walk(expr, trace, constraints)
    if MacroTools.@capture(expr, {symbol_} ~ dist_(args__))
        return quote
            let distribution = $(dist)($(args...)),
                value = haskey($constraints, $symbol) ? $constraints[$symbol] : rand(distribution),
                lp = logpdf(distribution, value)

                $trace[$symbol] = RV(value, lp)
                value
            end
        end
    elseif MacroTools.@capture(expr, @subppl func_(args__))
        return quote
            let (value, sub_trace) = $(func)(($(args...)), $constraints, $trace)
                value
            end
        end
    else
        return expr
    end
end


macro ppl(func)
    @assert MacroTools.@capture(func, (function f_(func_args__) body_ end))
    trace = gensym(:trace)
    constraints = gensym(:constraints)
    new_body = MacroTools.postwalk(ex -> walk(ex, trace, constraints), esc(body))
    return rmlines(quote
        function $(esc(f))($(esc.(func_args)...), $(esc(constraints))=Dict(), $(esc(trace))=Trace())
            function inner()
                $new_body
            end
            return inner(), $trace
        end
    end)
end

export @ppl, @subppl