
struct TraceTransformation
    f!::Function
    continuous_reads::Set{Any}
    continuous_writes::Set{Any}
    function TraceTransformation(f::Function)
        return new(f, Set{Any}(), Set{Any}())
    end
end

# bijective transformation of discrete variables does not need jacobian
# P(f(X) = x) = P(X = f^{-1}(x))
function read_discrete(tt::TraceTransformation, old_trace::Dict{Any,Real}, addr::Any)
    return old_trace[addr]
end

function write_discrete(tt::TraceTransformation, new_trace::Dict{Any,Real}, addr::Any, value::Real)
    new_trace[addr] = value
end

function read_continuous(tt::TraceTransformation, old_trace::Dict{Any,Real}, addr::Any)
    push!(tt.continuous_reads, addr)
    return old_trace[addr]
end

function write_continuous(tt::TraceTransformation, new_trace::Dict{Any,Real}, addr::Any, value::Real)
    push!(tt.continuous_writes, addr)
    new_trace[addr] = value
end

function copy_at_address(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real}, addr::Any)
    new_trace[addr] = old_trace[addr]
end

function copy_at_addresses(tt::TraceTransformation, old_trace::Dict{Any,Real}, old_addr::Any, new_trace::Dict{Any,Real}, new_addr::Any)
    new_trace[new_addr] = old_trace[old_addr]
end

function apply(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace = Dict{Any,Real}())
    empty!(tt.continuous_reads)
    empty!(tt.continuous_writes)
    tt.f!(tt, old_trace, new_trace)
    return new_trace
end

function apply(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}=Dict{Any,Real}(), new_proposal_trace::Dict{Any,Real}=Dict{Any,Real}())
    empty!(tt.continuous_reads)
    empty!(tt.continuous_writes)
    tt.f!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
    return new_model_trace, new_proposal_trace
end

function get_wrapped_f(tt::TraceTransformation, old_trace::Dict{Any,Real})
    function wrapped_f(X::AbstractVector{<:Real})
        # could be replace by directly reading from X at correct index ala JacobianPassState,
        # but is maybe not faster since we need to setup address_to_ix
        tracked_old_tr = Dict{Any,Real}()
        tracked_new_tr = Dict{Any,Real}()
        i = 0
        for (addr,v) in old_trace
            if addr in tt.continuous_reads
                i += 1
                tracked_old_tr[addr] = X[i]
            else
                tracked_old_tr[addr] = v
            end
        end
        @assert length(X) == i
        tt.f!(tt, tracked_old_tr, tracked_new_tr)
        # tracked_new_tr == new_trace
        return [v for (addr,v) in tracked_new_tr if addr in tt.continuous_writes]
    end
    continuous_input_arr = Float64[v for (addr, v) in old_trace if addr in tt.continuous_reads]
    return wrapped_f, continuous_input_arr
end

function get_wrapped_f(tt::TraceTransformation, old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real})
    function wrapped_f(X::AbstractVector{<:Real})
        # could be replace by directly reading from X at correct index ala JacobianPassState
        # but is maybe not faster since we need to setup address_to_ix
        tracked_old_model_tr = Dict{Any,Real}()
        tracked_old_proposal_tr = Dict{Any,Real}()
        tracked_new_model_tr = Dict{Any,Real}()
        tracked_new_proposal_tr = Dict{Any,Real}()
        i = 0
        for (addr,v) in old_model_trace
            if addr in tt.continuous_reads
                i += 1
                tracked_old_model_tr[addr] = X[i]
            else
                tracked_old_model_tr[addr] = v
            end
        end
        for (addr,v) in old_proposal_trace
            if addr in tt.continuous_reads
                i += 1
                tracked_old_proposal_tr[addr] = X[i]
            else
                tracked_old_proposal_tr[addr] = v
            end
        end
        @assert length(X) == i
        tt.f!(tt, tracked_old_model_tr, tracked_old_proposal_tr, tracked_new_model_tr, tracked_new_proposal_tr)
        return [v for (addr,v) in tracked_new_model_tr ∪ tracked_new_proposal_tr if addr in tt.continuous_writes]
    end
    continuous_input_arr = vcat(
        Float64[v for (addr, v) in old_model_trace if addr in tt.continuous_reads],
        Float64[v for (addr, v) in old_proposal_trace if addr in tt.continuous_reads],
    )
    return wrapped_f, continuous_input_arr
end

import Tracker

function jacobian_tracker(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real})
    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_trace)
    return Tracker.data(Tracker.jacobian(Tracker.collect ∘ wrapped_f, continuous_input_arr))
end

function jacobian_tracker(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}, new_proposal_trace::Dict{Any,Real})

    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_model_trace, old_proposal_trace)
    return Tracker.data(Tracker.jacobian(wrapped_f, continuous_input_arr))
end

import ForwardDiff

function jacobian_fwd_diff(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real})
    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_trace)

    continuous_input_arr = Float64[v for (addr, v) in old_trace if addr in tt.continuous_reads]
    return ForwardDiff.jacobian(wrapped_f, continuous_input_arr)
end

function jacobian_fwd_diff(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}, new_proposal_trace::Dict{Any,Real})

    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_model_trace, old_proposal_trace)
    return ForwardDiff.jacobian(wrapped_f, continuous_input_arr)
end

# ∂tt.f / ∂old_trace where f(old_trace) = new_trace
# transform has to be called before jacobian
function jacobian(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real})

    return jacobian_tracker(tt, old_trace, new_trace)
end

# ∂tt.f / ∂(old_model_trace, old_proposal_trace) where f(old_model_trace, old_proposal_trace) = (new_model_trace, new_proposal_trace)
# transform has to be called before jacobian
function jacobian(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}, new_proposal_trace::Dict{Any,Real})

    return jacobian_tracker(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
end


export TraceTransformation
export read_discrete, write_discrete
export read_continuous, write_continuous
export copy_at_address, copy_at_addresses
export apply
export jacobian