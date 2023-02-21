
import ..TinyPPL.Distributions: Proposal, logpdf

function lmh(pgm::PGM, n_samples::Int; proposal=Proposal())
    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)

    observed = .!isnothing.(pgm.observed_values)

    nodes = [n => [child for (x,child) in pgm.edges if x == n] for n in pgm.topological_order if !observed[n]]

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample(X) # initialise
    r = pgm.return_expr(X)

    n_accepted = 0 
    @progress for i in 1:n_samples
        node, children = rand(nodes)
        d = pgm.distributions[node](X)
        q = get(proposal, pgm.addresses[node], d)
        value_current = X[node]
        # lp_current = pgm.logpdf(X)
        W_current = logpdf(d, value_current)
        if !isempty(children)
            W_current += sum(logpdf(pgm.distributions[child](X), X[child]) for child in children)
        end
        value_proposed = rand(q)
        X[node] = value_proposed

        # lp_proposed = pgm.logpdf(X)
        W_proposed = logpdf(d, value_proposed)
        if !isempty(children)
            W_proposed += sum(logpdf(pgm.distributions[child](X), X[child]) for child in children)
        end
        
        log_α = W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        # log_α = lp_proposed - lp_current + logpdf(d, value_current) - logpdf(d, value_proposed)

        if log(rand()) < log_α
            n_accepted += 1
            r = pgm.return_expr(X)
        else
            X[node] = value_current
        end

        retvals[i] = r
        trace[:,i] = X
    end
    @info "LMH" n_accepted/n_samples

    return trace, retvals
end

export lmh