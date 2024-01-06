import LinearAlgebra: logabsdet

"""
Involutive MCMC
"""
function imcmc(model::UniversalModel, args::Tuple, observations::Dict,
    aux_model::UniversalModel, aux_args::Tuple,
    involution::Function,
    n_samples::Int; check_involution::Bool=false)

    transformation = TraceTransformation(involution)
    sampler = TraceSampler()

    # init
    model(args, sampler, observations)
    trace_current = sampler.X
    W_current = sampler.W

    traces = Vector{Dict{Any, Real}}(undef, n_samples)
    lp = Vector{Float64}(undef, n_samples)
    n_accepted = 0
    @progress for i in 1:n_samples
        sampler.W = 0.; sampler.X = Dict{Any,Real}()
        aux_model((trace_current, aux_args...), sampler, Dict())
        aux_current = sampler.X
        Q_current = sampler.W

        trace_proposed, aux_proposed = apply(transformation, trace_current, aux_current)
        J = jacobian_fwd_diff(transformation, trace_current, aux_current, trace_proposed, aux_proposed)

        if check_involution
            # TODO: check J dimension
            trace_current_2, aux_current_2 = apply(transformation, trace_proposed, aux_proposed)
            for addr in keys(trace_current) ∪ keys(trace_current_2)
                @assert trace_current[addr] ≈ trace_current_2[addr] (addr, trace_current[addr], trace_current_2[addr])
            end 
            for addr in keys(aux_current) ∪ keys(aux_current_2)
                @assert aux_current[addr] ≈ aux_current_2[addr] (addr, aux_current[addr], aux_current_2[addr])
            end
        end

        sampler.W = 0.; sampler.X = trace_proposed
        model(args, sampler, observations)
        W_proposed = sampler.W

        sampler.W = 0.; sampler.X = aux_proposed
        aux_model((trace_proposed, aux_args...), sampler, Dict())
        Q_proposed = sampler.W

        if log(rand()) < W_proposed + Q_proposed - W_current - Q_current + logabsdet(J)[1]
            n_accepted += 1
            trace_current = trace_proposed
            W_current = W_proposed
        end

        traces[i] = trace_current
        lp[i] = W_current
    end
    
    @info "RJMCMC" n_accepted/n_samples

    return UniversalTraces(traces), lp
end

export imcmc