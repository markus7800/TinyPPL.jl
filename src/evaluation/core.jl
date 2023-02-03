
import MacroTools

macro subppl(expr) return expr end # just a place holder

function walk(expr, sampler, constraints)
    if MacroTools.@capture(expr, {symbol_} ~ dist_(args__))
        return quote
            let distribution = $(esc(dist))($(args...)),
                obs = haskey($constraints, $symbol) ? $constraints[$symbol] : nothing,
                value = sample($sampler, $symbol, distribution, obs)
                value
            end
        end
    elseif MacroTools.@capture(expr, @subppl func_(args__))
        return quote
            let value = $(esc(func))(($(args...)), $sampler, $constraints)
                value
            end
        end
    else
        return expr
    end
end


macro ppl(func)
    @assert MacroTools.@capture(func, (function f_(func_args__) body_ end))
    sampler = gensym(:sampler)
    constraints = gensym(:constraints)
    new_body = MacroTools.postwalk(ex -> walk(ex, sampler, constraints), body)
    return rmlines(quote
        global function $(esc(f))($(func_args...), $sampler::Sampler, $constraints=Dict())
            function inner()
                $new_body
            end
            return inner()
        end
    end)
end

export @ppl, @subppl