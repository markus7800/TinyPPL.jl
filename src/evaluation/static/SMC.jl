
import Libtask
import Distributions: Categorical

# Assumptoin observe statements in static order

mutable struct SMCParticle <: StaticSampler
    log_γ::Float64 # log P(Y|X)
    trace::StaticTrace
    addresses_to_ix::Addr2Ix
end

function sample(sampler::SMCParticle, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        log_w = logpdf(dist, obs)
        sampler.log_γ += log_w
        Libtask.produce((addr, log_w))
        return obs
    end

    value = rand(dist) # could add proposal here
    sampler.trace[sampler.addresses_to_ix[addr]] = value
    
    return value
end

# Since sample contains only one produce we do not have to subtape sample
# Libtask.is_primitive(::typeof(sample), ::SMCParticle, args...) = false
get_retval(ttask::Libtask.TapedTask) = Libtask.result(ttask.tf)
has_returned(ttask::Libtask.TapedTask) = ttask.tf.retval_binding_slot != 0


function SMC(model::StaticModel, args::Tuple, observations::Observations, n_particles::Int)
    addresses_to_ix = get_address_to_ix(model, args, observations)

    # Heap-allocated Array and Ref objects are deep copied by default -> StaticTrace
    # Other heap-allocated objects (e.g., Dict) are shallow copied, by default -> Addr2Ix, model, etc.


    particles = [SMCParticle(0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix) for _ in 1:n_particles]
    tasks = [Libtask.TapedTask(model.f, args..., particles[i], observations) for i in 1:n_particles]

    traces = StaticTraces(addresses_to_ix, n_particles)
    logprobs = Vector{Float64}(undef, n_particles)

    addresses = Vector{Address}(undef, n_particles)
    log_w = Vector{Float64}(undef, n_particles)
    while true
        for i in 1:n_particles # TODO: parallelise
            consumed_value = Libtask.consume(tasks[i])
            if has_returned(tasks[i])
                traces.data[:,i] .= particles[i].trace
                traces.retvals[i] = get_retval(tasks[i])
                logprobs[i]  = particles[i].log_γ
                addresses[i] = :__BREAK
            else
                addresses[i], log_w[i] = consumed_value
            end
        end
        unique_addresses = unique(addresses)
        println(unique_addresses)
        @assert (length(unique_addresses) == 1) unique_addresses
        if addresses[1] == :__BREAK
            break
        end

        W = exp.(log_w)
        W = W / sum(W)
        # for resampling it does not matter if w / sum(w) (like in PMCMC and PGAS) or w⋅Z / sum(w⋅Z) (like in Intro to PPL)

        # A = rand(Categorical(W), n_particles)
        # _particles = copy(particles) # shallow copy to reorder
        # for i in 1:n_particles
        #     tasks[i] = copy(tasks[A[i]]) # fork
        #     particles[i] = deepcopy(_particles[A[i]])
        # end
    end

    return traces, normalise(logprobs)
end

export SMC
