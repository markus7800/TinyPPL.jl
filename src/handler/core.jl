
import ..TinyPPL.Distributions: Distribution, logpdf

abstract type Handler end

export Handler

const Message = Dict

export Message

const HANDLER_STACK = Handler[];

function enter(handler::Handler)
end

function exit(handler::Handler)
end

function process_message(handler::Handler, msg::Message)
end

function postprocess_message(handler::Handler, msg::Message)
end

function (handler::Handler)(args...)
    push!(HANDLER_STACK, handler)
    enter(handler)
    retval = handler.fn(args...)
    exit(handler)
    @assert pop!(HANDLER_STACK) == handler
    return retval
end

const Trace = Dict{Any, Message}
export Trace

mutable struct TraceHandler <: Handler
    fn::Union{Function,Handler}
    trace::Trace
    function TraceHandler(fn::Union{Function,Handler})
        this = new()
        this.fn = fn
        return this
    end
end
trace(fn::Union{Function,Handler}) = TraceHandler(fn)

export TraceHandler, trace

function enter(handler::TraceHandler)
    handler.trace = Trace()
end

function postprocess_message(handler::TraceHandler, msg::Message)
    @assert msg["type"] != "sample" || !haskey(handler.trace, msg["name"])
    handler.trace[msg["name"]] = copy(msg)
end

function get_trace(handler::TraceHandler, args...)
    retval = handler(args...)
    handler.trace[:RETURN] = Message("type"=>"return", "value"=>retval)
    return handler.trace
end

export get_trace

function logpdfsum(trace::Trace)::Float64
    lp = 0.
    for (_, msg) in trace
        if msg["type"] == "sample" || msg["type"] == "observation"
            lp += get!(msg, "logprob", logpdf(msg["fn"], msg["value"]))
        end
    end
    return lp
end

function logpdfsum(trace::Trace, filter_fn::Function)::Float64
    lp = 0.
    for (_, msg) in trace
        if (msg["type"] == "sample" || msg["type"] == "observation") && filter_fn(msg)
            lp += get!(msg, "logprob", logpdf(msg["fn"], msg["value"]))
        end
    end
    return lp
end

export logpdfsum

struct ReplayHandler <: Handler
    fn::Union{Function,Handler}
    guide_trace::Trace
end
replay(fn::Union{Function,Handler}, guide_trace::Trace) = ReplayHandler(fn, guide_trace)

export ReplayHandler, replay

function process_message(handler::ReplayHandler, msg::Message)
    if haskey(handler.guide_trace, msg["name"])
        msg["value"] = handler.guide_trace[msg["name"]]["value"]
    end
end 

struct BlockHandler <: Handler
    fn::Union{Function,Handler}
    filter_fn::Function
end
block(fn::Union{Function,Handler}, filter_fn::Function) = BlockHandler(fn, filter_fn)

export BlockHandler, block

function process_message(handler::BlockHandler, msg::Message)
    if !handler.filter_fn(msg)
        msg["stop"] = true
    end
end



function apply_stack!(msg::Message)
    # println(typeof.(HANDLER_STACK))
    pointer = 1
    for (i, handler) in enumerate(reverse(HANDLER_STACK))
        # println("process: ", typeof(handler))
        process_message(handler, msg)
        pointer = i
        if get(msg, "stop", false)
            break
        end
    end
    if isnothing(msg["value"])
        msg["value"] = msg["fn"](msg["args"]...)
    end
    for handler in HANDLER_STACK[end-pointer+1:end]
        # println("post_process: ", typeof(handler))
        postprocess_message(handler, msg)
    end
end

function (d::Distribution)(args...)
    return rand(d, args...)
end

function sample(name::Any, dist::Distribution, args...; obs=nothing)
    if isempty(HANDLER_STACK)
        return dist(args...)
    end

    msg = Message(
        "type" => isnothing(obs) ? "sample" : "observation",
        "name" => name,
        "fn" => dist,
        "args" => args,
        "value" => obs
    )
    apply_stack!(msg)
    return msg["value"]
end

export sample
