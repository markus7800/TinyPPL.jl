
function compile_lmh(pgm::PGM; static_observes::Bool=false, proposal=Proposal())
    X = gensym(:X)
    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)
    
    lmh_functions = Function[]
    for node in 1:pgm.n_variables
        !isnothing(pgm.observed_values[node]) && continue

        block_args = []
        value_current = gensym("value_current")
        push!(block_args, :($value_current = $X[$node]))

        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $(symbolic_dists[node])))

        children = [child for (x,child) in pgm.edges if x == node]
 
        log_α = gensym(:log_α) # W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        push!(block_args, :($log_α = 0.0))

        # compute W for current value
        for child in children
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(pgm.observed_values[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            push!(block_args, :($log_α -= logpdf($child_d_sym, $X[$child])))
        end

        # sample proposed value
        if haskey(proposal, pgm.addresses[node])
            q_sym = gensym("proposal_$node")
            q = proposal[pgm.addresses[node]]
            push!(block_args, :($q_sym = $(Expr(:call, typeof(q).name.name, params(q)...))))

            push!(block_args, :($log_α += logpdf($q_sym, $X[$node]) - logpdf($d_sym, $X[$node])))
            push!(block_args, :($X[$node] = rand($q_sym)))
            push!(block_args, :($log_α += logpdf($d_sym, $X[$node]) - logpdf($q_sym, $X[$node])))
        else
            # logpdf(d, value_proposed) - logpdf(d, value_current) +  logpdf(q, value_current) - logpdf(q, value_proposed) cancels
            push!(block_args, :($X[$node] = rand($d_sym)))
        end

        # compute W for proposed value
        for child in children
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(pgm.observed_values[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            push!(block_args, :($log_α += logpdf($child_d_sym, $X[$child])))          
        end

        # mh step
        push!(block_args, :(if log(rand()) < $log_α
                return true
            else
                $X[$node] = $value_current
                return false
            end)
        )  

        f_name = Symbol("$(pgm.name)_lmh_$node")
        f = rmlines(:(
            function $f_name($X::AbstractVector{Float64})
                $(Expr(:block, block_args...))
            end
        ))
        # display(f)
        f = eval(f)
        push!(lmh_functions, f)
    end

    X = Vector{Float64}(undef, pgm.n_variables);
    pgm.sample(X) # initialise
    @progress for f in lmh_functions
        println(f)
        Base.invokelatest(f, X)
    end
    return lmh_functions
end

function compiled_single_site(pgm::PGM, kernels::Vector{Function}, n_samples::Int; static_observes::Bool=false)

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample(X) # initialise
    r = pgm.return_expr(X)

    mask = isnothing.(pgm.observed_values)
    trace = Array{Float64,2}(undef, static_observes ? sum(mask) : pgm.n_variables, n_samples)
    retvals = Vector{Any}(undef, n_samples)

    n_accepted = 0 
    @progress for i in 1:n_samples
        k = rand(kernels)
        accepted = k(X)
        if accepted
            n_accepted += 1
            r = pgm.return_expr(X)
        end

        retvals[i] = r
        if static_observes
            trace[:,i] = X[mask]
        else  
            trace[:,i] = X
        end
    end
    @info "Compiled Single Site" n_accepted/n_samples

    return trace, retvals
end

function get_compute_W_block_args(pgm, plates, children, symbolic_dists, symbolic_observes, static_observes, X, log_α, W)
    block_args = []
    for child in children
        if child in plates
            plate_f_name = plate_function_name(pgm.name, :lp, child)
            if W == :current
                push!(block_args, :($log_α -= $plate_f_name($X)))
            else
                push!(block_args, :($log_α += $plate_f_name($X)))
            end
        else
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(symbolic_observes[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            if W == :current
                push!(block_args, :($log_α -= logpdf($child_d_sym, $X[$child])))
            else
                push!(block_args, :($log_α += logpdf($child_d_sym, $X[$child])))
            end
        end
    end
    return block_args
end

function compile_lmh(pgm::PGM, plate_symbols::Vector{Symbol}; static_observes::Bool=false, proposal=Proposal())
    X = gensym(:X)
    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)

    plates, plated_edges = pgm.plate_info.plates, pgm.plate_info.plated_edges

    lmh_functions = Function[]
    for node in 1:pgm.n_variables
        !isnothing(pgm.observed_values[node]) && continue

        block_args = []
        value_current = gensym("value_current")
        push!(block_args, :($value_current = $X[$node]))

        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $(symbolic_dists[node])))

        children = reduce(∪, [get_children(edge, node) for edge in plated_edges], init=[])
 
        log_α = gensym(:log_α) # W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        push!(block_args, :($log_α = 0.0))

        # compute W for current value
        append!(block_args, 
            get_compute_W_block_args(pgm, plates, children, symbolic_dists, symbolic_observes, static_observes, X, log_α, :current)
        )

        # sample proposed value
        if haskey(proposal, pgm.addresses[node])
            q_sym = gensym("proposal_$node")
            q = proposal[pgm.addresses[node]]
            push!(block_args, :($q_sym = $(Expr(:call, typeof(q).name.name, params(q)...))))

            push!(block_args, :($log_α += logpdf($q_sym, $X[$node]) - logpdf($d_sym, $X[$node])))
            push!(block_args, :($X[$node] = rand($q_sym)))
            push!(block_args, :($log_α += logpdf($d_sym, $X[$node]) - logpdf($q_sym, $X[$node])))
        else
            # logpdf(d, value_proposed) - logpdf(d, value_current) +  logpdf(q, value_current) - logpdf(q, value_proposed) cancels
            push!(block_args, :($X[$node] = rand($d_sym)))
        end

        # compute W for proposed value
        append!(block_args, 
            get_compute_W_block_args(pgm, plates, children, symbolic_dists, symbolic_observes, static_observes, X, log_α, :proposed)
        )

        # mh step
        push!(block_args, :(if log(rand()) < $log_α
                return true
            else
                $X[$node] = $value_current
                return false
            end)
        )  

        f_name = Symbol("$(pgm.name)_lmh_$node")
        f = rmlines(:(
            function $f_name($X::Vector{Float64})
                $(Expr(:block, block_args...))
            end
        ))
        # display(f)
        f = eval(f)
        push!(lmh_functions, f)
    end

    X = Vector{Float64}(undef, pgm.n_variables);
    pgm.sample(X) # initialise
    for f in lmh_functions
        # println(f)
        Base.invokelatest(f, X)
    end
    return lmh_functions
end

export compile_lmh, compiled_single_site