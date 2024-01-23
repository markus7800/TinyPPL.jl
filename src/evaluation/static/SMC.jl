
import Libtask
import Distributions: Categorical

# Assumptoin observe statements in static order

mutable struct SMCParticle <: StaticSampler
    t::Int
    log_γ::Float64 # log P(X,Y)
    log_Q::Float64 # log r(X_{t+1}|X_1:t,Y_1:t)
    trace::StaticTrace
    addresses_to_ix::Addr2Ix
    addr2proposal::Addr2Proposal
    score_only::Bool
end
SMCParticle(addresses_to_ix::Addr2Ix, addr2proposal::Addr2Proposal, score_only::Bool=false) = SMCParticle(0, 0., 0., Vector{Real}(undef,length(addresses_to_ix)), addresses_to_ix, addr2proposal, score_only)

Base.copy(p::SMCParticle) = SMCParticle(p.t, p.log_γ, 0., copy(p.trace), p.addresses_to_ix, p.addr2proposal, false) # sets log_Q to 0 and score_only to false

function sample(sampler::SMCParticle, addr::Address, dist::Distribution, obs::Union{Nothing,RVValue})::RVValue
    if !isnothing(obs)
        log_w = logpdf(dist, obs)
        sampler.log_γ += log_w
        Libtask.produce((addr, sampler.log_γ, sampler.log_Q))
        return obs
    end
    sampler.t += 1
    ix = sampler.addresses_to_ix[addr]
    @assert ix == sampler.t

    proposal_dist = get(sampler.addr2proposal, addr, StaticProposal(dist))
    if sampler.score_only
        value = sampler.trace[ix]
        lpq = proposal_logpdf(proposal_dist, value, (sampler.trace, sampler.addresses_to_ix))
    else # sample and score
        value, lpq = propose_and_logpdf(proposal_dist, (sampler.trace, sampler.addresses_to_ix))
        sampler.trace[ix] = value
    end
    sampler.log_Q += lpq
    sampler.log_γ += logpdf(dist, value)
    
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

function _smc_worker(model::StaticModel, args::Tuple, observations::Observations, logjoint::Function, addresses_to_ix::Addr2Ix, n_particles::Int,
    X_ref::Union{Nothing,StaticTrace}; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal(), check_addresses::Bool=false)

    conditional = !isnothing(X_ref)

    # particles[1] is reference particle
    particles = [SMCParticle(addresses_to_ix, addr2proposal) for _ in 1:n_particles]
    tasks = [Libtask.TapedTask(model.f, args..., particles[i], observations) for i in 1:n_particles]

    traces = StaticTraces(addresses_to_ix, n_particles)
    marginal_lik = 1

    addresses = Vector{Address}(undef, n_particles)
    log_w = Vector{Float64}(undef, n_particles)
    log_γ = zeros(n_particles)
    n_obs = 0
    while true
        if conditional
            # reference particle
            p_ref = particles[1]
            p_ref.score_only = true # all others should have score_only = false
            p_ref.trace[p_ref.t+1:end] = X_ref[p_ref.t+1:end]
            # x_t^i ~ Q(x_t | x_1:{t-1}) for i in  2:n_particles
            # x_t^1 = X_t
        end
        # else x_t^i ~ Q(x_t | x_1:{t-1}) for i in  1:n_particles

        for i in 1:n_particles # TODO: parallelise
            consumed_value = Libtask.consume(tasks[i])
            if has_returned(tasks[i])
                traces.data[:,i] .= particles[i].trace
                traces.retvals[i] = get_retval(tasks[i])
                addresses[i] = :__BREAK
            else
                addresses[i], log_γ_new, log_Q = consumed_value
                # compute W_t, W_1 = γ_1 / Q_1 is correctly computed since log_γ is initialised to 0.
                log_w[i] = log_γ_new - log_γ[i] - log_Q
                log_γ[i] = log_γ_new
            end
        end
        if check_addresses
            unique_addresses = unique(addresses)
            println(unique_addresses)
            @assert (length(unique_addresses) == 1) unique_addresses
        end

        # W_t
        # for resampling it does not matter if w / sum(w) (like in PMCMC and PGAS) or w⋅Z / sum(w⋅Z) (like in Intro to PPL)
        W = exp.(log_w) # = γ_t(x_1:t, y_1:t) / (γ_{t-1}(x_1:{t-1},y_1:{t-1}) * Q(X_t|x_1:{t-1},y_1:{t-1}))
        sum_W = sum(W)
        marginal_lik *= sum_W / n_particles
        W = W / sum_W

        if addresses[1] == :__BREAK
            break
        end
        n_obs += 1
        if n_obs == length(observations)
            # no resampling at last observation
            continue
        end

        # a_{t+1} ~ Categorical(W_t)
        A = rand(Categorical(W), n_particles)

        if conditional
            if ancestral_sampling
                # log_γ = log γ_t(x_1:t), log_w = log W_t(x_1:t) where x_1:t = particle.trace[1:particle.t]
                t = particles[1].t
                # this is very expensive and may be replaced by copying task and running particles to end
                log_γ_T = [logjoint(vcat(particles[i].trace[1:t], X_ref[t+1:end])) for i in 1:n_particles]
                log_w_tilde = normalise(log_w + log_γ_T - log_γ)
                W_tilde = exp.(log_w_tilde)
                A[1] = rand(Categorical(W_tilde / sum(W_tilde)))
            else
                A[1] = 1 # retain reference particle
            end
        end

        particles = copy.(particles[A]) # sets log_Q to 0 and score_only to false
        tasks = copy.(tasks[A]) # fork
        log_γ = log_γ[A]
        for i in 1:n_particles
            update_particle!(tasks[i], particles[i])
        end

        # println("Neff=", 1/sum(W.^2))
        # TODO: resample if collapse?
    end

    # TODO: handle sample is last
    logprobs = normalise(log_w)
    return traces, logprobs, marginal_lik
end


function smc(model::StaticModel, args::Tuple, observations::Observations, n_particles::Int; addr2proposal::Addr2Proposal=Addr2Proposal(), check_addresses::Bool=false)
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)

    return _smc_worker(model, args, observations, logjoint, addresses_to_ix, n_particles, nothing;
                       addr2proposal=addr2proposal, check_addresses=check_addresses)
end

export smc

function conditional_smc(model::StaticModel, args::Tuple, observations::Observations, n_particles::Int,
    X::StaticTrace; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal(), check_addresses::Bool=false)
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)

    return _smc_worker(model, args, observations, logjoint, addresses_to_ix, n_particles, X;
                       ancestral_sampling=ancestral_sampling, addr2proposal=addr2proposal, check_addresses=check_addresses)
end
export conditional_smc


function particle_gibbs(model::StaticModel, args::Tuple, observations::Observations, n_particles::Int, n_samples::Int; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal())
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)

    traces = StaticTraces(addresses_to_ix, n_samples)

    # initialise with SMC
    smc_traces, lps, _ = smc(model, args, observations, n_particles; addr2proposal=addr2proposal)
    W = exp.(lps)
    X = smc_traces[:, rand(Categorical(W))]

    @progress for i in 1:n_samples
        smc_traces, lps, _ = _smc_worker(model, args, observations, logjoint, addresses_to_ix, n_particles, X;
                                      ancestral_sampling=ancestral_sampling, addr2proposal=addr2proposal)
        W = exp.(lps)
        k = rand(Categorical(W))
        X = smc_traces[:,k]
        traces.data[:,i] = X
        traces.retvals[i] = smc_traces.retvals[k]
    end

    return traces
end

export particle_gibbs

# not really useful, but a sanity check if everyhing works.
function particle_IMH(model::StaticModel, args::Tuple, observations::Observations, n_particles::Int, n_samples::Int; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal())
    logjoint, addresses_to_ix = make_logjoint(model, args, observations)

    traces = StaticTraces(addresses_to_ix, n_samples)

    # initialise with SMC
    smc_traces, lps, marginal_lik = smc(model, args, observations, n_particles; addr2proposal=addr2proposal)
    W = exp.(lps)
    k = rand(Categorical(W))
    X_current = smc_traces[:, k]
    retval_current = smc_traces.retvals[k]
    marginal_lik_current = marginal_lik

    n_accept = 0
    @progress for i in 1:n_samples
        smc_traces, lps, marginal_lik_proposed = _smc_worker(
            model, args, observations, logjoint, addresses_to_ix, n_particles, nothing;
            ancestral_sampling=ancestral_sampling, addr2proposal=addr2proposal
        )
        W = exp.(lps)
        k = rand(Categorical(W))
        X_proposed = smc_traces[:,k]
        retval_proposed = smc_traces.retvals[k]

        if rand() < marginal_lik_proposed / marginal_lik_current
            n_accept += 1
            X_current = X_proposed
            retval_current = retval_proposed
            marginal_lik_current = marginal_lik_proposed
        end

        traces.data[:,i] = X_current
        traces.retvals[i] = retval_current
    end

    @info "particle IMH" n_accept/n_samples

    return traces
end

export particle_IMH