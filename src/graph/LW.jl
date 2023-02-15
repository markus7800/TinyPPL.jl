import ..TinyPPL.Distributions: logpdf

function likelihood_weighting(pgm::PGM, n_samples::Int)

    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)

    observed = .!isnothing.(pgm.observed_values)
    
    X = Vector{Float64}(undef, pgm.n_variables)
    @progress for i in 1:n_samples
        W = 0.
        for node in pgm.topological_order
            d = pgm.distributions[node](X)

            if observed[node]
                value = pgm.observed_values[node](X)
                W += logpdf(d, value)
            else
                value = rand(d)
            end
            X[node] = value
        end
        r = pgm.return_expr(X)

        logprobs[i] = W
        retvals[i] = r
        trace[:,i] = X
    end

    return trace, retvals, normalise(logprobs)
end

export likelihood_weighting

function get_symbolic_distributions(pgm::PGM, X::Symbol)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)

    symbolic_dists = []
    for node in 1:pgm.n_variables
        sym = ix_to_sym[node]
        d = pgm.symbolic_pgm.P[sym]
        for j in 1:pgm.n_variables
            d = substitute(ix_to_sym[j], :($X[$j]), d)
        end
        push!(symbolic_dists, d)
    end
    return symbolic_dists
end

function get_symbolic_observed_values(pgm::PGM, X::Symbol, static_observes::Bool)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)

    symbolic_observes = []
    for node in 1:pgm.n_variables
        sym = ix_to_sym[node]
        if isnothing(pgm.observed_values[node])
            push!(symbolic_observes, nothing)
        else
            y = pgm.symbolic_pgm.Y[sym]
            for j in 1:pgm.n_variables
                y = substitute(ix_to_sym[j], :($X[$j]), y)
            end
            if static_observes
                y = eval(y)
            end
            push!(symbolic_observes, y)
        end
    end
    return symbolic_observes
end

function compile_likelihood_weighting(pgm::PGM; static_observes::Bool=false)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)

    lp = gensym(:lp)
    block_args = []
    push!(block_args, :($lp = 0.0))

    X = gensym(:X)

    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)
    
    for i in pgm.topological_order
        sym = ix_to_sym[i]

        d_sym = gensym("dist_$i")
        push!(block_args, :($d_sym = $(symbolic_dists[i])))

        if haskey(pgm.symbolic_pgm.Y, sym)
            y = symbolic_observes[i]
            if static_observes
                push!(block_args, :($lp += logpdf($d_sym, $y)))
            else
                push!(block_args, :($X[$i] = $y))
                push!(block_args, :($lp += logpdf($d_sym, $X[$i])))
            end
        else
            push!(block_args, :($X[$i] = rand($d_sym)))
        end
    end

    new_E = deepcopy(pgm.symbolic_return_expr)
    for j in 1:pgm.n_variables
        new_E = substitute(ix_to_sym[j], :($X[$j]), new_E)
    end
    return_value = gensym(:return)
    push!(block_args, :($return_value = $new_E))

    push!(block_args, :($return_value, $lp))

    f_name = Symbol("$(pgm.name)_lw")
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $(Expr(:block, block_args...))
        end
    ))
    # display(f)
    lw = eval(f)
    X = Vector{Float64}(undef, pgm.n_variables); Base.invokelatest(lw, X); # compilation
    return lw
end

function compiled_likelihood_weighting(pgm::PGM, lw::Function, n_samples::Int; static_observes::Bool=false)
    X = Vector{Float64}(undef, pgm.n_variables)
    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    mask = isnothing.(pgm.observed_values)
    trace = Array{Float64,2}(undef, static_observes ? sum(mask) : pgm.n_variables, n_samples)

    @progress for i in 1:n_samples
        retvals[i], logprobs[i] = lw(X)
        if static_observes
            trace[:,i] = X[mask]
        else
            trace[:,i] = X
        end
    end
    
    return trace, retvals, normalise(logprobs)
end

export compile_likelihood_weighting, compiled_likelihood_weighting
