
import ..TinyPPL.Distributions: Proposal, Addr2Var, logpdf, random_walk_proposal_dist

function lmh(pgm::PGM, n_samples::Int; proposal=Proposal())
    function kernel(X::Vector{Float64}, node::Int)
        d = pgm.distributions[node](X)
        q = get(proposal, pgm.addresses[node], d)

        value_current = X[node]
        value_proposed = rand(q)
        X[node] = value_proposed

        Q_difference = logpdf(q, value_current) - logpdf(d, value_current)
        Q_difference += logpdf(d, value_proposed) - logpdf(q, value_proposed)

        return Q_difference
    end
    return single_site(pgm, n_samples, kernel)
end

function rwmh(pgm::PGM, n_samples::Int; addr2var=Addr2Var(), default_var::Float64=1.)
    function kernel(X::Vector{Float64}, node::Int)
        d = pgm.distributions[node](X)
        var = get(addr2var, pgm.addresses[node], default_var)

        value_current = X[node]
        forward_dist = random_walk_proposal_dist(d, value_current, var)
        value_proposed = rand(forward_dist)
        X[node] = value_proposed
        backward_dist = random_walk_proposal_dist(d, value_proposed, var)

        Q_difference = logpdf(backward_dist, value_current) - logpdf(d, value_current)
        Q_difference += logpdf(d, value_proposed) - logpdf(forward_dist, value_proposed)

        return Q_difference
    end
    return single_site(pgm, n_samples, kernel)
end

function single_site(pgm::PGM, n_samples::Int, kernel::Function)
    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)

    observed = .!isnothing.(pgm.observed_values)

    nodes = [n => [child for (x,child) in pgm.edges if x == n] for n in pgm.topological_order if !observed[n]]

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample!(X) # initialise
    r = pgm.return_expr(X)

    n_accepted = 0 
    @progress for i in 1:n_samples
        node, children = rand(nodes)
        value_current = X[node]
        W_current = 0.0
        if !isempty(children)
            W_current += sum(logpdf(pgm.distributions[child](X), X[child]) for child in children)
        end

        Q_difference = kernel(X, node) 

        W_proposed = 0.0
        if !isempty(children)
            W_proposed += sum(logpdf(pgm.distributions[child](X), X[child]) for child in children)
        end
        
        log_α = W_proposed - W_current + Q_difference

        if log(rand()) < log_α
            n_accepted += 1
            r = pgm.return_expr(X)
        else
            X[node] = value_current
        end

        retvals[i] = r
        trace[:,i] = X
    end
    @info "Single Site" n_accepted/n_samples

    return trace, retvals
end

export lmh, rwmh