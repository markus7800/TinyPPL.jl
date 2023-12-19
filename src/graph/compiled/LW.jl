
function compile_likelihood_weighting(pgm::PGM; static_observes::Bool=false)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)

    lp = gensym(:lp)
    block_args = []
    push!(block_args, :($lp = 0.0))

    X = gensym(:X)

    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)
    
    if isnothing(pgm.plate_info)
        ordered_nodes = pgm.topological_order
    else
        ordered_nodes = get_topolocial_order(pgm.n_variables, pgm.plate_info)
    end

    d_sym = gensym("dist")
    for node in ordered_nodes
        if node isa Plate
            if all(!isnothing(symbolic_observes[child]) for child in node.nodes)
                if !static_observes
                    plate_f_name = plate_function_name(pgm.name, :sample, node)
                    push!(block_args, :($plate_f_name($X)))
                end
                plate_f_name = plate_function_name(pgm.name, :lp, node)
                push!(block_args, :($lp += $plate_f_name($X)))
            else
                @assert all(isnothing(symbolic_observes[child]) for child in node.nodes)
                plate_f_name = plate_function_name(pgm.name, :sample, node)
                push!(block_args, :($plate_f_name($X)))
            end
        else
            push!(block_args, :($d_sym = $(symbolic_dists[node])))
            if !isnothing(symbolic_observes[node])
                y = symbolic_observes[node]
                if static_observes
                    push!(block_args, :($lp += logpdf($d_sym, $y)))
                else
                    push!(block_args, :($X[$node] = $y))
                    push!(block_args, :($lp += logpdf($d_sym, $X[$node])))
                end
            else
                push!(block_args, :($X[$node] = rand($d_sym)))
            end
        end
    end
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:pgm.n_variables)
    new_E = subtitute_for_syms(var_to_expr, deepcopy(pgm.symbolic_return_expr), X)

    return_value = gensym(:return)
    push!(block_args, :($return_value = $new_E))

    push!(block_args, :($return_value, $lp))

    f_name = Symbol("$(pgm.name)_lw")
    f = rmlines(:(
        function $f_name($X::Vector{Float64})
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

    if static_observes
        pgm.sample!(X)
    end

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
