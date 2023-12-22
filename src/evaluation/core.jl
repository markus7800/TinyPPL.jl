
import MacroTools

macro subppl(expr) return expr end # just a place holder

function walk_ppl_sytnax(expr, sampler, constraints)
    if MacroTools.@capture(expr, {symbol_} ~ dist_)
        return quote
            let distribution = $dist,
                obs = haskey($constraints, $symbol) ? $constraints[$symbol] : nothing,
                value = sample($sampler, $symbol, distribution, obs)
                value
            end
        end
    elseif MacroTools.@capture(expr, {symbol_} ~ dist_(args__))
        println("Am I redundant?") # TODO
        return quote
            let distribution = $(dist)($(args...)),
                obs = haskey($constraints, $symbol) ? $constraints[$symbol] : nothing,
                value = sample($sampler, $symbol, distribution, obs)
                value
            end
        end
    elseif MacroTools.@capture(expr, param(symbol_, args__))
        return quote
            param($sampler, $symbol, $(args...))
        end
    elseif MacroTools.@capture(expr, @subppl func_(args__))
        return quote
            let value = $(func).f(($(args...)), $sampler, $constraints)
                value
            end
        end
    else
        return expr
    end
end


macro ppl(annotations, func)
    if annotations isa Symbol
        annotations = Set{Symbol}([annotations])
    else
        annotations = Set{Symbol}(annotations.args)
    end
    return ppl_macro(annotations, func)
end

macro ppl(func)
    return ppl_macro(Set{Symbol}(), func)
end

abstract type Model end
struct UniversalModel
    f::Function
end
function (model::UniversalModel)(args::Tuple, sampler::UniversalSampler, constraints::Dict)
    return model.f(args..., sampler, constraints)
end

# for all executions, the same (finite number of) random variables
# with same distribution + support
struct StaticModel
    f::Function
end
function (model::StaticModel)(args::Tuple, sampler::StaticSampler, constraints::Dict)
    return model.f(args..., sampler, constraints)
end


function ppl_macro(annotations::Set{Symbol}, func)
    @assert MacroTools.@capture(func, (function f_(func_args__) body_ end))
    sampler = gensym(:sampler)
    constraints = gensym(:constraints)
    # implicit addressing, escaping body is needed as it could be referencing symbols not defined in body
    new_body = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ ~ dist_) && !isbraced(var) ? :($var = {$(QuoteNode(var))} ~ $dist) : expr, esc(body));
    new_body = MacroTools.postwalk(ex -> walk_ppl_sytnax(ex, sampler, constraints), new_body)

    sampler_type = :static in annotations ? StaticSampler : UniversalSampler
    model_type = :static in annotations ? StaticModel : UniversalModel

    julia_fn_name = gensym(f)
    return rmlines(quote
        function $(esc(julia_fn_name))($(esc.(func_args)...), $(esc(sampler))::$(esc(sampler_type)), $(esc(constraints))=Dict())
            function inner()
                $new_body
            end
            return inner()
        end
        $(esc(f)) = $(esc(model_type))($(esc(julia_fn_name)))
    end)
end

export @ppl, @subppl