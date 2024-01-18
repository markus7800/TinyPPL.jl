
import Libtask
import Distributions: Categorical

# Assumptoin observe statements in static order

mutable struct SMCParticle <: StaticSampler
    log_γ::Float64 # log P(X,Y)
    log_Q::Float64 # log r(X_{t+1}|X_1:t,Y_1:t)
    trace::StaticTrace
    addresses_to_ix::Addr2Ix
    addr2proposal::Addr2Proposal
end
SMCParticle(addresses_to_ix::Addr2Ix, addr2proposal::Addr2Proposal) = SMCParticle(0., 0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix, addr2proposal)

Base.copy(p::SMCParticle) = SMCParticle(p.log_γ, 0., copy(p.trace), p.addresses_to_ix, p.addr2proposal) # sets log_Q to 0

function sample(sampler::SMCParticle, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        log_w = logpdf(dist, obs)
        sampler.log_γ += log_w
        Libtask.produce((addr, sampler.log_γ, sampler.log_Q))
        return obs
    end

    proposal_dist = get(sampler.addr2proposal, addr, StaticProposal(dist))
    value, lpq = propose_and_logpdf(proposal_dist, (sampler.trace, sampler.addresses_to_ix))
    sampler.log_Q += lpq
    sampler.log_γ += logpdf(dist, value)
    sampler.trace[sampler.addresses_to_ix[addr]] = value
    
    return value
end

# Since sample contains only one produce we do not have to subtape sample
# Libtask.is_primitive(::typeof(sample), ::SMCParticle, args...) = false
get_retval(ttask::Libtask.TapedTask) = Libtask.result(ttask.tf)
has_returned(ttask::Libtask.TapedTask) = ttask.tf.retval_binding_slot != 0

# manually switch particle in arg binding slot in taped function
function update_particle!(tf::Libtask.TapedFunction, particle::SMCParticle)
    sampler_slot = tf.arity - 1 + 1 # [func, args..., sampler, observations]
    @assert Libtask._arg(tf, sampler_slot) isa SMCParticle
    Libtask._arg!(tf, sampler_slot, particle)
    for sub_tf in values(tf.subtapes)
        update_particle!(sub_tf, particle)
    end
end
update_particle!(task::Libtask.TapedTask, particle::SMCParticle) = update_particle!(task.tf, particle)

# Following Particle Gibbs Ancestral Sampling
function smc(model::StaticModel, args::Tuple, observations::Observations, n_particles::Int; addr2proposal::Addr2Proposal=Addr2Proposal())
    addresses_to_ix = get_address_to_ix(model, args, observations)

    particles = [SMCParticle(addresses_to_ix, addr2proposal) for _ in 1:n_particles]
    tasks = [Libtask.TapedTask(model.f, args..., particles[i], observations) for i in 1:n_particles]

    traces = StaticTraces(addresses_to_ix, n_particles)

    addresses = Vector{Address}(undef, n_particles)
    log_w = Vector{Float64}(undef, n_particles)
    log_γ = zeros(n_particles)
    while true
        for i in 1:n_particles # TODO: parallelise
            consumed_value = Libtask.consume(tasks[i])
            if has_returned(tasks[i])
                traces.data[:,i] .= particles[i].trace
                traces.retvals[i] = get_retval(tasks[i])
                addresses[i] = :__BREAK
            else
                # addresses[i], log_w[i] = consumed_value
                addresses[i], log_γ_new, log_Q = consumed_value
                log_w[i] = log_γ_new - log_γ[i] - log_Q
                log_γ[i] = log_γ_new
            end
        end
        unique_addresses = unique(addresses)
        println(unique_addresses)
        @assert (length(unique_addresses) == 1) unique_addresses
        if addresses[1] == :__BREAK
            break
        end

        W = exp.(log_w) # = γ_t(X_1:t, Y_1:t) / (γ_{t-1}(X_1:{t-1},Y_1:{t-1}) * P(X_t|X_1:{t-1},Y_1:{t-1})) = p(Y_t | X_1:t, Y_1:{t-1})
        W = W / sum(W)
        # for resampling it does not matter if w / sum(w) (like in PMCMC and PGAS) or w⋅Z / sum(w⋅Z) (like in Intro to PPL)

        A = rand(Categorical(W), n_particles)
        particles = copy.(particles[A]) # sets log_Q to 0
        tasks = copy.(tasks[A]) # fork
        log_γ = log_γ[A]
        for i in 1:n_particles
            update_particle!(tasks[i], particles[i])
        end

        println("Neff=", 1/sum(W.^2))
        # TODO: resample if collapse?
    end

    logprobs = normalise(log_w)
    return traces, logprobs
end

export smc
