
function compiled_single_site(pgm::PGM, kernels::Vector{Function}, n_samples::Int)
    X = Vector{Float64}(undef, pgm.n_latents)
    pgm.sample!(X) # initialise
    r = get_retval(pgm, X)

    trace = Array{Float64,2}(undef, pgm.n_latents, n_samples)
    retvals = Vector{Any}(undef, n_samples)

    n_accepted = 0
    @progress for i in 1:n_samples
        k = rand(kernels) # select update kernel at random
        accepted = k(X, pgm.observations)
        if accepted
            n_accepted += 1
            r = get_retval(pgm, X)
        end

        retvals[i] = r
        trace[:,i] = X
    end
    @info "Compiled Single Site" n_accepted/n_samples

    return GraphTraces(pgm, trace, retvals)
end

"""
Builds function that only computes W_current / W_proposed by only accumulating relevant factors (children)
"""
function get_compute_W_block_args(pgm::PGM, plates::Vector{Plate}, children::Vector, symbolic_dists::Vector, X::Symbol, Y::Symbol, log_α::Symbol, W::Symbol)
    op = W == :current ? :(-=) : :(+=)
    block_args = []
    for child in children
        if child in plates
            plate_f_name = plate_function_name(pgm.name, :lp, child)
            push!(block_args, Expr(op, log_α, :($plate_f_name($X, $Y))))
        else
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if isobserved(pgm, child)
                push!(block_args, Expr(op, log_α, :(logpdf($child_d_sym, $Y[$child - $(pgm.n_latents)]))))
            else
                push!(block_args, Expr(op, log_α, :(logpdf($child_d_sym, $X[$child]))))
            end
        end
    end
    return block_args
end

function compile_lmh(pgm::PGM; addr2Proposal::Addr2Proposal=Addr2Proposal())
    function lmh_kernel(block_args, symbolic_dists, node, X, log_α)
        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $(symbolic_dists[node])))

        if haskey(addr2Proposal, pgm.addresses[node])
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
    end
    return compile_single_site(pgm, lmh_kernel)
end

function get_rw_dist_expr(d_sym, symbol_dist, value, var)
    if symbol_dist.head == :call
        dist_name = symbol_dist.args[1]
        # println(dist_name)
        if dist_name == :Bernoulli
            return :(Bernoulli(1 - $value))
        elseif dist_name in [:Geometric, :Poisson]
            return :(DiscreteRWProposer(0, Inf, Int($value), $var))
        elseif dist_name == :Binomial
            n = symbol_dist.args[2]
            return :(DiscreteRWProposer(0, $n, Int($value), $var))
        elseif dist_name == :Categorical
            return :(DiscreteRWProposer(1, ncategories($d_sym), Int($value), $var))
        elseif dist_name == :DiscreteUniform
            a, b = symbol_dist.args[2:3]
            return :(DiscreteRWProposer(Int($a), Int($b), Int($value), $var))
        elseif dist_name in [:Exponential, :Gamma, :InverseGamma, :LogNormal]     
            return :(ContinuousRWProposer(0, Inf, $value, $var))
        elseif dist_name in [:Cauchy, :Laplace, :Normal, :TDist]      
            return :(Normal($value, sqrt($var)))  
        elseif dist_name == :Beta
            return :(ContinuousRWProposer(0., 1., $value, $var))
        elseif dist_name == :Uniform
            a, b = symbol_dist.args[2:3]
            return :(ContinuousRWProposer($a, $b, $value, $var))
        end
    end
    return :(random_walk_proposal_dist($d_sym, $value, $var))
end

function compile_rwmh(pgm::PGM; addr2var=Addr2Var(), default_var::Float64=1.)
    function rwmh_kernel(block_args, symbolic_dists, node, X, log_α)
        dist = symbolic_dists[node]
        d_sym = gensym("dist_$node")
        push!(block_args, :($d_sym = $dist))

        var = get(addr2var, pgm.addresses[node], default_var)

        forward_d_sym = gensym("forward_dist")
        value_current = gensym("value_current")
        backward_d_sym = gensym("backward_dist")
        value_proposed = gensym("value_proposed")


        push!(block_args, :($value_current = $X[$node]))
        push!(block_args, :($forward_d_sym = $(get_rw_dist_expr(d_sym, dist, value_current, var))))
        push!(block_args, :($value_proposed = rand($forward_d_sym)))
        push!(block_args, :($X[$node] = $value_proposed))
        push!(block_args, :($backward_d_sym = $(get_rw_dist_expr(d_sym, dist, value_proposed, var))))

        push!(block_args, :($log_α += logpdf($backward_d_sym, $value_current) - logpdf($d_sym, $value_current)))
        push!(block_args, :($log_α += logpdf($d_sym, $value_proposed) - logpdf($forward_d_sym, $value_proposed)))
    end
    return compile_single_site(pgm, rwmh_kernel)
end

"""
For each latent variable, builds a function that implements a MH update for thi variable.
Alternative implementation: All in one function and goto statements to go to correct variable based on resample variable.
"""
function compile_single_site(pgm::PGM, kernel::Function)
    X = gensym(:X)
    Y = gensym(:Y)

    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:pgm.n_latents)
    symbolic_dists = get_symbolic_distributions(pgm.symbolic_pgm, pgm.n_variables, ix_to_sym, var_to_expr, X)

    if isnothing(pgm.plate_info)
        plates, plated_edges = Plate[], nothing
    else
        plates, plated_edges = pgm.plate_info.plates, pgm.plate_info.plated_edges
    end

    # compile one update function for each variable
    lmh_functions = Function[]
    value_current = gensym("value_current")
    for node in 1:pgm.n_latents

        block_args = []
        push!(block_args, :($value_current = $X[$node]))

        if isnothing(plated_edges)
            children = [child for (x,child) in pgm.edges if x == node]
        else
            children = reduce(∪, [get_children(edge, node) for edge in plated_edges], init=[])
        end
 
        log_α = gensym(:log_α) # W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)
        push!(block_args, :($log_α = 0.0))

        # compute W for current value
        append!(block_args, 
            get_compute_W_block_args(pgm, plates, children, symbolic_dists, X, Y, log_α, :current)
        )

        # sample proposed value
        kernel(block_args, symbolic_dists, node, X, log_α)

        # compute W for proposed value
        append!(block_args, 
            get_compute_W_block_args(pgm, plates, children, symbolic_dists, X, Y, log_α, :proposed)
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
            function $f_name($X::Vector{Float64}, $Y::Vector{Float64})
                $(Expr(:block, block_args...))
            end
        ))
        # display(f)
        f = eval(f)
        push!(lmh_functions, f)
    end

    X = Vector{Float64}(undef, pgm.n_latents);
    pgm.sample!(X) # initialise
    for f in lmh_functions
        # println(f)
        Base.invokelatest(f, X, pgm.observations)
    end
    return lmh_functions
end

export compile_lmh, compile_rwmh, compiled_single_site