import Distributions: logpdf

function likelihood_weighting(pgm::PGM, n_samples::Int)

    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)
    
    X = Vector{Float64}(undef, pgm.n_variables)
    @progress for i in 1:n_samples
        W = 0.
        for node in pgm.topological_order
            d = pgm.distributions[node](X)

            if !isnothing(pgm.observed_values[node])
                value = pgm.observed_values[node](X)
                X[node] = value
                W += logpdf(d, value)
            else
                value = rand(d)
                X[node] = value
            end
        end
        r = pgm.return_expr(X)

        @inbounds logprobs[i] = W
        @inbounds retvals[i] = r
        @inbounds trace[:,i] .= X
        
    end

    return trace, retvals, normalise(logprobs)
end

export likelihood_weighting


function compile_likelihood_weighting(pgm::PGM)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)

    lp = gensym(:lp)
    block_args = []
    push!(block_args, :($lp = 0.0))

    X = gensym(:X)
    for i in pgm.topological_order
        sym = ix_to_sym[i]
        d = pgm.symbolic_pgm.P[sym]
        for j in 1:pgm.n_variables
            d = substitute(ix_to_sym[j], :($X[$j]), d)
        end

        d_sym = gensym("dist_$i")
        push!(block_args, :($d_sym = $d))

        if haskey(pgm.symbolic_pgm.Y, sym)
            y = pgm.symbolic_pgm.Y[sym]
            for j in 1:pgm.n_variables
                y = substitute(ix_to_sym[j], :($X[$j]), y)
            end
            push!(block_args, :($X[$i] = $y))
            push!(block_args, :($lp += logpdf($d_sym, $X[$i])))
        else
            push!(block_args, :($X[$i] = rand($d)))
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
    return lw
end

function compiled_likelihood_weighting(pgm::PGM, lw::Function, n_samples::Int)

    retvals = Vector{Any}(undef, n_samples)
    logprobs = Vector{Float64}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)
    
    X = Vector{Float64}(undef, pgm.n_variables)
    @progress for i in 1:n_samples
        r, W = lw(X)
        @inbounds logprobs[i] = W
        @inbounds retvals[i] = r
        @inbounds trace[:,i] .= X
    end
    return trace, retvals, normalise(logprobs)
end

export compile_likelihood_weighting, compiled_likelihood_weighting