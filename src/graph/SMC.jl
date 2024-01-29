import ..TinyPPL.Distributions: logpdf

# simple and lightweight implementation
function light_smc(pgm::PGM, n_particles::Int)    
    particles = [Vector{Float64}(undef, pgm.n_latents) for _ in 1:n_particles]
    log_w = Vector{Float64}(undef, n_particles)
    marginal_lik = 1

    @progress for node in pgm.topological_order
        if isobserved(pgm, node)
            for i in 1:n_particles
                X = particles[i]
                d = get_distribution(pgm, node, X)
                value = get_observed_value(pgm, node)
                log_w[i] = logpdf(d, value)
            end

            # resampling
            W = exp.(log_w)
            sum_W = sum(W)
            marginal_lik *= sum_W / n_particles
            W = W / sum_W

            A = rand(Categorical(W), n_particles)
            particles = copy.(particles[A])
        else
            for i in 1:n_particles
                X = particles[i]
                d = get_distribution(pgm, node, X)
                X[node] = rand(d)
            end
        end
    end
    retvals = Vector{Any}(undef, n_particles)
    for i in 1:n_particles
        X = particles[i]
        retvals[i] = get_retval(pgm, X)
    end
    traces = reduce(hcat, particles)
    return GraphTraces(pgm, traces, retvals), normalise(log_w), marginal_lik
end
export light_smc

# Same structure as in static/SMC.jl
# Instead of using Libtask, we exploit the fixed structure of a PGM.
# We traverse the topological_order for several particles at once.
# Additionally, we can optimise ancestral_sampling by not computing the full joint,
# but only those factors that are influenced by the current state.
function _smc_worker(pgm::PGM, n_particles::Int, X_ref::Union{Nothing,Vector{Float64}};
    ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal(),
    relevant_future_nodes::Union{Nothing, Vector{Vector{Int}}} = nothing
    )
    conditional = !isnothing(X_ref)
    last_observe_node = first(Iterators.reverse(Iterators.filter(node -> isobserved(pgm, node), pgm.topological_order)))

    particles = [Vector{Float64}(undef, pgm.n_latents) for _ in 1:n_particles]
    log_w = Vector{Float64}(undef, n_particles)
    log_w_tilde = Vector{Float64}(undef, n_particles)
    log_q = zeros(n_particles)
    log_γ = zeros(n_particles)
    log_γ_new = zeros(n_particles)
    marginal_lik = 1

    for (t, node) in enumerate(pgm.topological_order)
        if isobserved(pgm, node)
            for i in 1:n_particles
                X = particles[i]
                d = get_distribution(pgm, node, X)
                value = get_observed_value(pgm, node)
                log_γ_new[i] += logpdf(d, value)

                log_w[i] = log_γ_new[i] - log_γ[i] - log_q[i]
                # TODO: handle sample is last
            end
            W = exp.(log_w)
            sum_W = sum(W)
            marginal_lik *= sum_W / n_particles
            W = W / sum_W

            # println("Neff=", 1/sum(W.^2))

            if node == last_observe_node
                # no resampling at last observe
                continue
            end

            # resampling
            A = rand(Categorical(W), n_particles)
            if conditional
                if ancestral_sampling
                    for i in 1:n_particles
                        X = particles[i]
                        if isnothing(relevant_future_nodes)
                            for future_node in pgm.topological_order[t+1:end]
                                if !isobserved(pgm, future_node)
                                    X[future_node] = X_ref[future_node]
                                end
                            end
                            # this takes long because of dispatch
                            # log_γ_future = sum(
                            #     logpdf(get_distribution(pgm, future_node, X), get_value(pgm, future_node, X))
                            #     for future_node in pgm.topological_order[t+1:end]; init=0.
                            # )
                            # log_w_tilde[i] = log_w[i] + log_γ_future

                            # @assert log_γ_new[i] + log_γ_future ≈ pgm.logpdf(X, pgm.observations)

                            # this is faster
                            log_w_tilde[i] = log_w[i] + pgm.logpdf(X, pgm.observations) - log_γ_new[i]
                        else
                            # optimisation:
                            # γ(x_1:T) / γ(x_1:t) = p(xref_t+1:T,y_t+1:T | x:1_t,y_1:t) = p(xref_{relevant_future_nodes[t]} | x:1_t,y_1:t) * C(xref)
                            # e.g. for x_1 → x_2 → ... → x_T
                            # p(xref_t+1:T,y_t+1:T | x:1_t,y_1:t) = p(xref_t+1 | x_t) * p(xref_t+2 | xref_t+1) ...
                            # relevant_future_nodes[t] = {t+1}
                            # relevant_future_nodes is computed once in particle_gibbs
                            log_γ_future_relevant = sum(
                                logpdf(get_distribution(pgm, future_node, X), X_ref[future_node])
                                for future_node in relevant_future_nodes[t]; init=0.
                            )
                            log_w_tilde[i] = log_w[i] + log_γ_future_relevant
                        end
                    end
                    W_tilde = exp.(normalise(log_w_tilde))
                    A[1] = rand(Categorical(W_tilde / sum(W_tilde)))
                else
                    A[1] = 1 # retain reference particle
                end
            end

            particles = copy.(particles[A])
            log_γ = log_γ_new[A]
            log_γ_new .= log_γ
            log_q .= 0
        else
            if conditional
                # reference particle
                X = particles[1]
                d = get_distribution(pgm, node, X)
                addr = get_address(pgm, node)
                proposal_dist = get(addr2proposal, addr, StaticProposal(d))

                value = X_ref[node]
                lpq = proposal_logpdf(proposal_dist, value, (addr,X))
                log_q[1] += lpq
                log_γ_new[1] += logpdf(d, value)

                X[node] = value

                start = 2
            else
                start = 1
            end
            for i in start:n_particles
                X = particles[i]
                d = get_distribution(pgm, node, X)
                addr = get_address(pgm, node)
                proposal_dist = get(addr2proposal, addr, StaticProposal(d))

                value, lpq = propose_and_logpdf(proposal_dist, (addr,X))
                log_q[i] += lpq
                log_γ_new[i] += logpdf(d, value)

                X[node] = value
            end
        end
    end

    retvals = Vector{Any}(undef, n_particles)
    for i in 1:n_particles
        X = particles[i]
        retvals[i] = get_retval(pgm, X)
    end
    traces = reduce(hcat, particles)
    return GraphTraces(pgm, traces, retvals), normalise(log_w), marginal_lik
end

function smc(pgm::PGM, n_particles::Int; addr2proposal::Addr2Proposal=Addr2Proposal())
    return _smc_worker(pgm, n_particles, nothing; addr2proposal=addr2proposal)
end

export smc


function conditional_smc(pgm::PGM, n_particles::Int, X_ref::Vector{Float64}; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal())
    return _smc_worker(pgm, n_particles, X_ref; ancestral_sampling=ancestral_sampling, addr2proposal=addr2proposal)
end

export conditional_smc

function get_relevant_future_nodes(pgm::PGM)
    T = length(pgm.topological_order)
    relevant_future_nodes = [Int[] for _ in 1:T]
    for t in 1:T
        for future_node in pgm.topological_order[t+1:end]
            # better to traverse topological_order backwards
            if any((past_node => future_node) in pgm.edges for past_node in Iterators.reverse(pgm.topological_order[1:t]))
                push!(relevant_future_nodes[t], future_node)
            end
        end
    end
    return relevant_future_nodes
end

function particle_gibbs(pgm::PGM, n_particles::Int, n_samples::Int; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal(), init=:SMC)

    pg_traces = Array{Float64}(undef, pgm.n_latents, n_samples)
    retvals = Vector{Any}(undef, n_samples)

    relevant_future_nodes = ancestral_sampling ? get_relevant_future_nodes(pgm) : nothing

    if init == :SMC
        # initialise with SMC
        smc_traces, lps, _ = smc(pgm, n_particles; addr2proposal=addr2proposal)
        W = exp.(lps)
        X = smc_traces[:, rand(Categorical(W))]
    else
        X = zeros(pgm.n_latents)
    end

    @progress for i in 1:n_samples
        smc_traces, lps, _ = _smc_worker(pgm, n_particles, X;
                                      ancestral_sampling=ancestral_sampling, addr2proposal=addr2proposal,
                                      relevant_future_nodes=relevant_future_nodes)
        W = exp.(lps)
        k = rand(Categorical(W))
        X = smc_traces[:,k]
        pg_traces[:,i] = X
        retvals[i] = get_retval(pgm, X)
    end

    return GraphTraces(pgm, pg_traces, retvals)
end

export particle_gibbs

# not really useful, but a sanity check if everyhing works.
function particle_IMH(pgm::PGM, n_particles::Int, n_samples::Int; ancestral_sampling::Bool=false, addr2proposal::Addr2Proposal=Addr2Proposal())

    pg_traces = Array{Float64}(undef, pgm.n_latents, n_samples)
    retvals = Vector{Any}(undef, n_samples)

    # initialise with SMC
    smc_traces, lps, marginal_lik = smc(pgm, n_particles; addr2proposal=addr2proposal)
    W = exp.(lps)
    k = rand(Categorical(W))
    X_current = smc_traces[:, k]
    retval_current = smc_traces.retvals[k]
    marginal_lik_current = marginal_lik

    n_accept = 0
    @progress for i in 1:n_samples
        smc_traces, lps, marginal_lik_proposed = _smc_worker(
            pgm, n_particles, nothing;
            addr2proposal=addr2proposal
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

        pg_traces[:,i] = X_current
        retvals[i] = retval_current
    end

    @info "particle IMH" n_accept/n_samples

    return GraphTraces(pgm, pg_traces, retvals)
end

export particle_IMH