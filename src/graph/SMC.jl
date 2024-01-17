import ..TinyPPL.Distributions: logpdf

function smc(pgm::PGM, n_particles::Int)
    observed = .!isnothing.(pgm.observed_values)
    
    particles = [Vector{Float64}(undef, pgm.n_variables) for _ in 1:n_particles]
    log_w = Vector{Float64}(undef, n_particles)

    @progress for node in pgm.topological_order
        if observed[node]
            for i in 1:n_particles
                X = particles[i]
                d = pgm.distributions[node](X)
                value = pgm.observed_values[node](X)
                log_w[i] = logpdf(d, value)
                X[node] = value
            end
            W = exp.(log_w)
            W = W / sum(W)

            A = rand(Categorical(W), n_particles)
            _particles = copy(particles) # shallow copy to reorder
            for i in 1:n_particles
                particles[i] = copy(_particles[A[i]])
            end
        else
            for i in 1:n_particles
                X = particles[i]
                d = pgm.distributions[node](X)
                X[node] = rand(d)
            end
        end
    end
    traces = reduce(hcat, particles)
    return traces, normalise(log_w)
end

export smc
