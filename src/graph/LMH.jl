
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
    return has_lazy_latents(pgm) ? lazy_single_site(pgm, n_samples, kernel) : single_site(pgm, n_samples, kernel)
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
    return has_lazy_latents(pgm) ? lazy_single_site(pgm, n_samples, kernel) : single_site(pgm, n_samples, kernel)
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

Does not support models with `lazy_ifs`, see lazy_single_site.
"""
function single_site(pgm::PGM, n_samples::Int, kernel::Function)
    if has_lazy_latents(pgm)
        error("Model has latent variables with Flat distribution (lazy_ifs), which single_site does not support. Use lazy_single_site.")
    end

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

contains_symbol(expr::Expr, sym::Symbol) = any(contains_symbol(arg, sym) for arg in expr.args)
contains_symbol(expr::Symbol, sym::Symbol) = expr == sym
contains_symbol(expr, sym::Symbol) = false

# true if the model was compiled with lazy_ifs and has latent variables in branches,
# i.e. latent variables with distribution `Flat` if their branch is not taken.
# (observed variables in branches also have distribution `Flat`, which does not need special treatment)
function has_lazy_latents(pgm::PGM)
    spgm = pgm.symbolic_pgm
    return any(contains_symbol(d, :Flat) for (v, d) in spgm.P if !haskey(spgm.Y, v))
end

"""
Single-site Metropolis Hastings for models compiled with `lazy_ifs`.

Latent variables in branches that are not taken have distribution `Flat` and value NaN (inactive).
Updating node `n` may change which branches are taken and thus (de)activate variables.
Activated variables are sampled from their distribution and deactivated variables are set to NaN.

Since the branch conditions accumulate for nested ifs and every use of a variable in a branch goes through the if expression,
all variables that may be (de)activated by changing `n` (and all variables that depend on them) are children of `n`.
Processing the children of `n` in topological order thus handles nested ifs
(the distribution of a child is evaluated with the updated values of the children before it).

For an activated variable z, log p(X',Y') contains log p(z'|...) and log Q(X'|X) contains the same term (prior proposal),
so they cancel. Similarly, for a deactivated variable, log p(z|...) cancels with the term of the reverse proposal in log Q(X|X').
Thus, we only have to evaluate the children of `n` that are not (de)activated.

Selecting an inactive variable does not change the model and is a no-op move.
"""
function lazy_single_site(pgm::PGM, n_samples::Int, kernel::Function)
    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_latents, n_samples)

    # handle nested ifs with topological order:
    # the distribution of a child is evaluated with the updated (new) values of the children before it
    topological_position = zeros(Int, pgm.n_variables)
    for (i, n) in enumerate(pgm.topological_order)
        topological_position[n] = i
    end
    nodes = [n => sort([child for (x,child) in pgm.edges if x == n], by=child -> topological_position[child]) for n in 1:pgm.n_latents]

    X = Vector{Float64}(undef, pgm.n_latents)
    pgm.sample!(X) # initialise
    r = get_retval(pgm, X)

    n_accepted = 0 
    @progress for i in 1:n_samples
        node, children = rand(nodes)
        if isnan(X[node])
            # inactive variable, does not affect the model
            retvals[i] = r
            trace[:,i] = X
            continue
        end

        value_current = X[node]
        W_current = [logpdf(get_distribution(pgm, child, X), get_value(pgm, child, X)) for child in children]

        log_α = kernel(X, node) # Q_resample_address, sets X[node] to proposed value

        toggled = Pair{Int,Float64}[] # child => value_current
        for (j, child) in enumerate(children)
            d = get_distribution(pgm, child, X)
            if child <= pgm.n_latents && isnan(X[child]) != (d isa Flat)
                # (de)activated: sample from prior (or set to NaN = rand(Flat())), prior and proposal terms cancel
                # store old value for rejection
                push!(toggled, child => X[child])
                X[child] = rand(d)
            else
                log_α += logpdf(d, get_value(pgm, child, X)) - W_current[j]
            end
        end

        if log(rand()) < log_α
            n_accepted += 1
            r = get_retval(pgm, X)
        else
            X[node] = value_current
            for (child, value) in toggled
                X[child] = value
            end
        end

        retvals[i] = r
        trace[:,i] = X
    end
    @info "Lazy Single Site" n_accepted/n_samples

    return GraphTraces(pgm, trace, retvals)
end

export lmh, rwmh