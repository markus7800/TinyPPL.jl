
import MacroTools
import Libtask
import TinyPPL: rmlines, isbraced

macro subppl(expr) return expr end # just a place holder

# The model body is escaped (it is user code), and so is the code generated here.
# Hygiene: local variables get fresh names (gensym), such that they do not capture user variables
# (e.g. in `{:x => obs} ~ d`) and functions are referenced with GlobalRef, such that they do not have
# to be imported in (or are not shadowed by) the module where @ppl is used.
const SAMPLE = GlobalRef(@__MODULE__, :sample)
const PARAM = GlobalRef(@__MODULE__, :param)
const GET = GlobalRef(Base, :get)

function walk_ppl_sytnax(expr, sampler, observations)
    if MacroTools.@capture(expr, {symbol_} ~ dist_)
        distribution, obs = gensym(:distribution), gensym(:obs)
        # the address expression is not bound to a variable, since Libtask (static SMC) cannot handle
        # IR statements that are just a constant (e.g. `%1 = :x` for `let addr = :x`)
        return quote
            let $distribution = $dist, $obs = $GET($observations, $symbol, nothing)
                $SAMPLE($sampler, $symbol, $distribution, $obs)
            end
        end
    elseif MacroTools.@capture(expr, param(symbol_, args__))
        return quote
            $PARAM($sampler, $symbol, $(args...))
        end
    elseif MacroTools.@capture(expr, @subppl func_(args__))
        return quote
            $(func).f($(args...), $sampler, $observations)
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

# no assumptions about program
struct UniversalModel <: Model
    f::Function
end
function (model::UniversalModel)(args::Tuple, sampler::UniversalSampler, observations::Observations)
    return model.f(args..., sampler, observations)
end

# for all executions, the same (finite number of) random variables will be instantiated
struct StaticModel <: Model
    f::Function
end
function (model::StaticModel)(args::Tuple, sampler::StaticSampler, observations::Observations)
    return model.f(args..., sampler, observations)
end

to_universal(model::StaticModel) = UniversalModel(model.f)
export to_universal

"""
@ppl function model_name(args...)
    body with sample or parameter statements
end

sample syntax: `{address} ~ distribution`
sugar: `address ~ distribution` is equivalent to `address = {:address} ~ distribution`

parameter syntax: p = param(name; size=1, constraint=NoConstraint())
submodel syntax: @subppl submodel(args...)

returns an UniversalModel model_name which is callable with
    model_name(args::Tuple, sampler::UniversalSampler, observations::Observations)
if the funtion is annotated with @ppl static function model_name(...
then the macro generates a StaticModel which is callable with
    model_name(args::Tuple, sampler::StaticSampler, observations::Observations)
where we assume that for all executions, the same (finite number of) random variables will be instantiated
"""
function ppl_macro(annotations::Set{Symbol}, func)
    @assert MacroTools.@capture(func, (function f_(func_args__) body_ end))
    sampler = gensym(:sampler)
    observations = gensym(:observations)
    # implicit addressing, escaping body is needed as it could be referencing symbols not defined in body
    new_body = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ ~ dist_) && !isbraced(var) ? :($var = {$(QuoteNode(var))} ~ $dist) : expr, esc(body));
    new_body = MacroTools.postwalk(ex -> walk_ppl_sytnax(ex, sampler, observations), new_body)

    sampler_type = Sampler # :static in annotations ? StaticSampler : UniversalSampler
    model_type = :static in annotations ? StaticModel : UniversalModel

    julia_fn_name = gensym(f)
    return rmlines(quote
        function $(esc(julia_fn_name))($(esc.(func_args)...), $(esc(sampler))::$(esc(sampler_type)), $(esc(observations))=Dict())
            $new_body
        end
        $(esc(f)) = $(esc(model_type))($(esc(julia_fn_name)))
        Libtask.is_primitive(::typeof($(esc(julia_fn_name))), args...) = false
    end)
end

export @ppl, @subppl