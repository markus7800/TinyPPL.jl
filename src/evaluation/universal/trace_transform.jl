
# == almost Guide sampler # TODO: replace GuideSampler?
mutable struct TraceSampler <: UniversalSampler
    W::Real # depends on the eltype of phi and X
    # params_to_ix::Param2Ix
    # phi::AbstractVector{<:Real}
    X::Dict{Any,Real}
    # constraints::Dict{Any,ParamConstraint}
    function TraceSampler(; X::Dict{Any,Real}=Dict{Any,Real}())#, params_to_ix::Param2Ix=Param2Ix(), phi::AbstractVector{<:Real}=Float64[], constraints=Dict{Any,ParamConstraint}())
        return new(0., X)
    end
end

function sample(sampler::TraceSampler, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        # difference here
        sampler.W += logpdf(dist, obs)
        return obs
    end
    # evaluate at given value or sample and store
    value = get!(sampler.X, addr, rand(dist))
    sampler.W += logpdf(dist, value)
    return value
end

# currently not used
# function param(sampler::TraceSampler, addr::Any, size::Int=1, constraint::ParamConstraint=Unconstrained())
#     # keeps track of all parameters and tracks them if initial sampler.phi is tracked
#     if !haskey(sampler.params_to_ix, addr)
#         n = length(sampler.phi)
#         ix = (n+1):(n+size)
#         sampler.params_to_ix[addr] = ix
#         sampler.constraints[addr] = constraint
#         # all parameters are initialised to 0
#         if Tracker.istracked(sampler.phi)
#             sampler.phi = vcat(sampler.phi, Tracker.param(zeros(size)))
#         else
#             sampler.phi = vcat(sampler.phi, zeros(eltype(sampler.phi), size))
#         end
#     end
#     ix = sampler.params_to_ix[addr]
#     if size == 1
#         return transform(constraint, sampler.phi[ix[1]])
#     else
#         return transform(constraint, sampler.phi[ix])
#     end
# end

struct TraceTransformation
    f!::Function
    continuous_reads::Set{Any}
    continuous_writes::Set{Any}
    function TraceTransformation(f::Function)
        return new(f, Set{Any}(), Set{Any}())
    end
end

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
export copy_at_address
export apply
export jacobian