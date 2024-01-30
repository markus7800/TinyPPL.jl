
import ..TinyPPL.Distributions: Addr2Proposal, Addr2Var, logpdf, random_walk_proposal_dist

"""
Lightweight Metropolis Hastings

See single_site for MH step.

The user can define the conditional update for the resample address Q(x_proposed | x_current)
by providing a ProposalDistribution with Addr2Proposal.
Otherwise, by default we unconditionally propose from the program prior.
"""
function lmh(pgm::PGM, n_samples::Int; addr2proposal=Addr2Proposal())
    function kernel(X::Vector{Float64}, node::Int)
        d = pgm.distributions[node](X)
        q = get(addr2proposal, get_address(pgm, node), StaticProposal(d))

        value_current = X[node]

        value_proposed, forward_logpdf = propose_and_logpdf(q, value_current)
        backward_logpdf = proposal_logpdf(q, value_current, value_proposed)

        # in single_site we only accumulate logpdf of children
        Q_resample_address = backward_logpdf - logpdf(d, value_current)
        Q_resample_address += logpdf(d, value_proposed) - forward_logpdf

        X[node] = value_proposed

        return Q_resample_address
    end
    return single_site(pgm, n_samples, kernel)
end

"""
Random Walk Metropolis Hastings

See single_site for MH step.

This is equivalent to running lmh with setting every proposal distribution
to the corresponding RandomWalkProposal competible with addr2var
"""
function rwmh(pgm::PGM, n_samples::Int; addr2var=Addr2Var(), default_var::Float64=1.)
    function kernel(X::Vector{Float64}, node::Int)
        d = pgm.distributions[node](X)
        var = get(addr2var, pgm.addresses[node], default_var)

        value_current = X[node]
        forward_dist = random_walk_proposal_dist(d, value_current, var)
        value_proposed = rand(forward_dist)
        X[node] = value_proposed
        backward_dist = random_walk_proposal_dist(d, value_proposed, var)

        # in single_site we only accumulate logpdf of children
        Q_resample_address = logpdf(backward_dist, value_current) - logpdf(d, value_current)
        Q_resample_address += logpdf(d, value_proposed) - logpdf(forward_dist, value_proposed)

        return Q_resample_address
    end
    return single_site(pgm, n_samples, kernel)
end

"""
Single-site Metropolis Hastings

Idea: change value (resample) of only at one address in current trace
Since we have a static trace we do not have to consider changing addresses.

The acceptance probability is given by

log α = log p(X',Y') + log Q(X|X') - log p(X,Y) - log Q(X'|X), where

log Q(X'|X) = log(1/|X|) + log Q(x_proposed | x_current) + sum(log Q[addr] for addr in addresses(X') - addresses(X))

For PGMs, this simplifies to 

log α = log p(X',Y') + log Q(x_current | x_proposed) - log p(X,Y) - log Q(X' | x_current)

Also if we update node `n`, we know that this affects only the contribution to log p(X,Y) of `n` and its children. 
"""
function single_site(pgm::PGM, n_samples::Int, kernel::Function)
    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_latents, n_samples)

    nodes = [n => [child for (x,child) in pgm.edges if x == n] for n in 1:pgm.n_latents]

    X = Vector{Float64}(undef, pgm.n_latents)
    Y = pgm.observations
    pgm.sample!(X) # initialise
    r = get_retval(pgm, X)

    n_accepted = 0 
    @progress for i in 1:n_samples
        node, children = rand(nodes)
        value_current = X[node]
        W_current = sum(logpdf(get_distribution(pgm, child, X), get_value(pgm, child, X)) for child in children; init=0.)

        Q_resample_address = kernel(X, node) 

        W_proposed = sum(logpdf(get_distribution(pgm, child, X), get_value(pgm, child, X)) for child in children; init=0.)
        
        log_α = W_proposed - W_current + Q_resample_address

        if log(rand()) < log_α
            n_accepted += 1
            r = get_retval(pgm, X)
        else
            X[node] = value_current
        end

        retvals[i] = r
        trace[:,i] = X
    end
    @info "Single Site" n_accepted/n_samples

    return GraphTraces(pgm, trace, retvals)
end

export lmh, rwmh