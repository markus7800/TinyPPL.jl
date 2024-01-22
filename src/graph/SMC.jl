import ..TinyPPL.Distributions: logpdf

function smc(pgm::PGM, n_particles::Int)    
    particles = [Vector{Float64}(undef, pgm.n_latents) for _ in 1:n_particles]
    log_w = Vector{Float64}(undef, n_particles)

    @progress for node in pgm.topological_order
        if isobserved(pgm, node)
            for i in 1:n_particles
                X = particles[i]
                d = get_distribution(pgm, node, X)
                value = get_observed_value(pgm, node)
                log_w[i] = logpdf(d, value)
            end
            W = exp.(log_w)
            W = W / sum(W)

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
    return GraphTraces(pgm, traces, retvals), normalise(log_w)
end

export smc
