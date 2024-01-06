"""
Wrapper for an inplace bijective deterministic transformation from UniversalTrace -> UniversalTrace
or UniversalTrace × UniversalTrace -> UniversalTrace × UniversalTrace
Computes the Jacobian of this transformation with AD.
This is useful if you have a log density d over UniversalTrace and want to transform it with
the density transformation rule log p(T(X)) = log d(X) - log abs det ∇T(X)
in more common notation: for Y = T(X) we have log p(Y) =log  d(T^{-1}(Y)) + log abs det ∇T^{-1}(Y)

To compute log abs det ∇T we only have to take the jacobian of continuous variables,
since for discrete variables p(T(X) = x) = p(X = T^{-1}(x)).
Also if some continuous variables are copied then we can ignore them since they correspond to
a row-column in the jacobian which one 1 and 0 else, which would just flip the sign of ∇T.
Also the order of variables does not mather as they would also just flip the sign.

To automatically compute the jacobian, you have to *first* call `apply`, which records all
continuous variables, and then call `jacobian` on the result.

f! should satisfy length(continuous_reads) = length(continuous_writes) to be a diffeomorphism.
Distributions are supported on some manifold and a diffeomorphism preserves the dimension.

Addresses of proposals are assumed to be disjoint from model addresses.
"""
struct TraceTransformation
    # f!:: (tt:TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace) or
    # f!:: (tt:TraceTransformation, old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace, new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace) or
    # f! has to read/write/copy values from old to new with the interface below
    f!::Function
    continuous_reads::Set{Address}  # tracks which continuous variables are read from old_trace / (old_model_trace, old_proposal_trace)
    continuous_writes::Set{Address} # tracks which continuous variables are writeten to new_trace / (new_model_trace, new_proposal_trace)
    function TraceTransformation(f!::Function)
        return new(f!, Set{Address}(), Set{Address}())
    end
end
Base.show(io::IO, tt::TraceTransformation) = print(io, "TraceTransformation($(tt.f!))")

# bijective transformation of discrete variables does not need jacobian
function read_discrete(tt::TraceTransformation, old_trace::AbstractUniversalTrace, addr::Address)
    return old_trace[addr]
end

function write_discrete(tt::TraceTransformation, new_trace::AbstractUniversalTrace, addr::Address, value::RVValue)
    new_trace[addr] = value
end

# copied variables do not need jacobian
function copy_at_address(tt::TraceTransformation, old_trace::AbstractUniversalTrace, new_trace::AbstractUniversalTrace, addr::Address)
    new_trace[addr] = old_trace[addr]
end

function copy_at_addresses(tt::TraceTransformation, old_trace::AbstractUniversalTrace, old_addr::Address, new_trace::AbstractUniversalTrace, new_addr::Address)
    new_trace[new_addr] = old_trace[old_addr]
end

function read_continuous(tt::TraceTransformation, old_trace::AbstractUniversalTrace, addr::Address)
    push!(tt.continuous_reads, addr)
    return old_trace[addr]
end

function write_continuous(tt::TraceTransformation, new_trace::AbstractUniversalTrace, addr::Address, value::RVValue)
    push!(tt.continuous_writes, addr)
    new_trace[addr] = value
end

function apply!(tt::TraceTransformation, old_trace::AbstractUniversalTrace, new_trace::AbstractUniversalTrace)
    empty!(tt.continuous_reads)
    empty!(tt.continuous_writes)
    tt.f!(tt, old_trace, new_trace)
    return new_trace
end

function apply(tt::TraceTransformation, old_trace::AbstractUniversalTrace)
    return apply!(tt, old_trace, UniversalTrace())
end

function apply!(tt::TraceTransformation,
    old_model_trace::AbstractUniversalTrace, old_proposal_trace::AbstractUniversalTrace,
    new_model_trace::AbstractUniversalTrace, new_proposal_trace::AbstractUniversalTrace)
    empty!(tt.continuous_reads)
    empty!(tt.continuous_writes)
    tt.f!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
    return new_model_trace, new_proposal_trace
end

function apply(tt::TraceTransformation, old_model_trace::AbstractUniversalTrace, old_proposal_trace::AbstractUniversalTrace)
    return apply!(tt, old_model_trace, old_proposal_trace, UniversalTrace(), UniversalTrace())
end


# wrap transformation as function which maps vectors to vectors for AD
# this is a bit hacky: Dict -> Vector -> Dict
# Could be replaced with Dictionaries.jl where convertion is light weight.
function get_wrapped_f(tt::TraceTransformation, old_trace::AbstractUniversalTrace)
    function wrapped_f(X::AbstractVector{<:Real})
        # could be replace by directly reading from X at correct index ala JacobianPassState,
        # but is maybe not faster since we need to setup address_to_ix
        tracked_old_tr = UniversalTrace()
        tracked_new_tr = UniversalTrace()
        i = 0
        for (addr,v) in old_trace
            if addr in tt.continuous_reads
                i += 1
                tracked_old_tr[addr] = X[i]     # note: change here for multivariate support
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

# wrap transformation as function which maps vectors to vectors for AD
function get_wrapped_f(tt::TraceTransformation, old_model_trace::AbstractUniversalTrace, old_proposal_trace::AbstractUniversalTrace)
    function wrapped_f(X::AbstractVector{<:Real})
        # could be replace by directly reading from X at correct index ala JacobianPassState
        # but is maybe not faster since we need to setup address_to_ix
        tracked_old_model_tr = UniversalTrace()
        tracked_old_proposal_tr = UniversalTrace()
        tracked_new_model_tr = UniversalTrace()
        tracked_new_proposal_tr = UniversalTrace()
        i = 0
        for (addr,v) in old_model_trace
            if addr in tt.continuous_reads
                i += 1
                tracked_old_model_tr[addr] = X[i]       # note: change here for multivariate support
            else
                tracked_old_model_tr[addr] = v
            end
        end
        for (addr,v) in old_proposal_trace
            if addr in tt.continuous_reads
                i += 1
                tracked_old_proposal_tr[addr] = X[i]    # note: change here for multivariate support
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

function jacobian_tracker(tt::TraceTransformation, old_trace::AbstractUniversalTrace, new_trace::AbstractUniversalTrace)
    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_trace)
    return Tracker.data(Tracker.jacobian(Tracker.collect ∘ wrapped_f, continuous_input_arr))
end

function jacobian_tracker(tt::TraceTransformation,
    old_model_trace::AbstractUniversalTrace, old_proposal_trace::AbstractUniversalTrace,
    new_model_trace::AbstractUniversalTrace, new_proposal_trace::AbstractUniversalTrace)

    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_model_trace, old_proposal_trace)
    return Tracker.data(Tracker.jacobian(wrapped_f, continuous_input_arr))
end

import ForwardDiff

function jacobian_fwd_diff(tt::TraceTransformation, old_trace::AbstractUniversalTrace, new_trace::AbstractUniversalTrace)
    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_trace)

    continuous_input_arr = Float64[v for (addr, v) in old_trace if addr in tt.continuous_reads]
    return ForwardDiff.jacobian(wrapped_f, continuous_input_arr)
end

function jacobian_fwd_diff(tt::TraceTransformation,
    old_model_trace::AbstractUniversalTrace, old_proposal_trace::AbstractUniversalTrace,
    new_model_trace::AbstractUniversalTrace, new_proposal_trace::AbstractUniversalTrace)

    wrapped_f, continuous_input_arr = get_wrapped_f(tt, old_model_trace, old_proposal_trace)
    return ForwardDiff.jacobian(wrapped_f, continuous_input_arr)
end

# ∂tt.f / ∂old_trace where f(old_trace) = new_trace
# transform has to be called before jacobian
function jacobian(tt::TraceTransformation, old_trace::AbstractUniversalTrace, new_trace::AbstractUniversalTrace)
    # include new_trace as argument to force call of apply before

    return jacobian_fwd_diff(tt, old_trace, new_trace)
end

# ∂tt.f / ∂(old_model_trace, old_proposal_trace) where f(old_model_trace, old_proposal_trace) = (new_model_trace, new_proposal_trace)
# transform has to be called before jacobian
function jacobian(tt::TraceTransformation,
    old_model_trace::AbstractUniversalTrace, old_proposal_trace::AbstractUniversalTrace,
    new_model_trace::AbstractUniversalTrace, new_proposal_trace::AbstractUniversalTrace)
    # include new_model_trace, new_proposal_trace as argument to force call of apply before

    return jacobian_fwd_diff(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
end


export TraceTransformation
export read_discrete, write_discrete
export read_continuous, write_continuous
export copy_at_address, copy_at_addresses
export apply
export jacobian