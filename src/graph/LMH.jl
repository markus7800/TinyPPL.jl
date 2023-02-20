
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
        W_current = sum(logpdf(pgm.distributions[child](X), X[child]) for child in children) + logpdf(d, value_current)

        value_proposed = rand(q)
        X[node] = value_proposed

        # lp_proposed = pgm.logpdf(X)
        W_proposed = sum(logpdf(pgm.distributions[child](X), X[child]) for child in children) + logpdf(d, value_proposed)
        
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

struct Plate
    symbol::Symbol
    nodes::UnitRange
end

abstract type PlatedEdge end
struct PlateToPlateEdge <: PlatedEdge
    from::Plate
    to::Plate
end
function get_children(edge::PlateToPlateEdge, node::Int)
    if node in edge.from.nodes
        return [edge.to] # return plate
    end
    return Int[]
end

struct NodeToPlateEdge <: PlatedEdge
    from::Int
    to::Plate
end
function get_children(edge::NodeToPlateEdge, node::Int)
    if edge.from == node
        return [edge.to] # return plate
    end
    return Int[]
end

struct PlateToNodeEdge <: PlatedEdge
    from::Plate
    to::Int
end
function get_children(edge::PlateToNodeEdge, node::Int)
    if node in edge.from.nodes
        return Int[edge.to]
    end
    return Int[]
end

struct InterPlateEdge <: PlatedEdge
    from::Plate
    to::Plate
    bijection::Set{Pair{Int,Int}}
end
function get_children(edge::InterPlateEdge, node::Int)
    if node in edge.from.nodes
        return [e[2] for e in edge.bijection if e[1]==node] # should be exactly one edge
    end
    return Int[]
end

struct NodeToNodeEdge <: PlatedEdge
    from::Int
    to::Int
end
function get_children(edge::NodeToNodeEdge, node::Int)
    if edge.from == node
        return [edge.to]
    end
    return Int[]
end

function plate_transformation(pgm::PGM, plate_symbols::Vector{Symbol})
    # addresses of pgm are sort by plate
    plates = Plate[]
    for plate_symbol in plate_symbols
        i = pgm.n_variables
        j = 0
        nodes = Vector{Int}()
        for (node, addr) in enumerate(pgm.addresses)
            if addr isa Pair && addr[1] == plate_symbol
                push!(nodes, node)
                i = min(i, node)
                j = max(j, node)
            end
        end
        @assert nodes == collect(i:j)
        push!(plates, Plate(plate_symbol, i:j))
    end
    plated_edges = Set{PlatedEdge}()
    edges = deepcopy(pgm.edges)

    for node in 1:pgm.n_variables
        for plate in plates
            if all((node=>plate_node) in edges for plate_node in plate.nodes)
                # plate depends on node
                for plate_node in plate.nodes
                    delete!(edges, node=>plate_node)
                end
                push!(plated_edges, NodeToPlateEdge(node, plate))
            end
            if all((plate_node=>node) in edges for plate_node in plate.nodes)
                # node depends on plate
                for plate_node in plate.nodes
                    delete!(edges, plate_node=>node)
                end
                # println(plate, "->", node)
                push!(plated_edges, PlateToNodeEdge(plate, node))
            end
        end
    end
    for plate in plates, other_plate in plates
        if all(NodeToPlateEdge(plate_node, other_plate) in plated_edges for plate_node in plate.nodes)
            for plate_node in plate.nodes
                delete!(plated_edges, NodeToPlateEdge(plate_node, other_plate))
            end
            push!(plated_edges, PlateToPlateEdge(plate, other_plate))
        end
        if all(PlateToNodeEdge(other_plate, plate_node) in plated_edges for plate_node in plate.nodes)
            for plate_node in plate.nodes
                delete!(plated_edges, PlateToNodeEdge(other_plate, plate_node))
            end
            push!(plated_edges, PlateToPlateEdge(other_plate, plate))
        end
    end

    for plate in plates, other_plate in plates
        # find bijections
        plate_nodes = plate.nodes
        other_plate_nodes = other_plate.nodes
        if length(plate_nodes) != length(other_plate_nodes)
            continue
        end
        bijection = Set{Pair{Int, Int}}()
        is_bijection = true
        for plate_node in plate_nodes
            found = false
            for e in edges
                if e[1] == plate_node && e[2] in other_plate_nodes
                    if found # assert only one edge for each plate node
                        is_bijection = false
                    end
                    push!(bijection, e)
                    found = true
                end
            end
        end
        is_bijection &= length(bijection) == length(plate_nodes) # assert all plate nodes have edge
        if is_bijection
            for e in bijection
                delete!(edges, e)
            end
            push!(plated_edges, InterPlateEdge(plate, other_plate, bijection))
        end            
    end
    
    for e in edges
        push!(plated_edges, NodeToNodeEdge(e[1], e[2]))
    end

    pgm, plates, plated_edges
end

function compile_lmh(pgm::PGM, plate_symbols::Vector{Symbol}; static_observes::Bool=false, proposal=Proposal())
    pgm, plates, plated_edges = plate_transformation(pgm, plate_symbols);
    X = gensym(:X)
    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)

    # setup functions to compute lp for plates
    plate_functions = Function[]
    plate_function_names = Dict{Plate, Symbol}()
    for plate in plates
        block_args = []
        lp = gensym(:lp)
        push!(block_args, :($lp = 0.0))

        for child in plate.nodes
            child_d_sym = gensym("child_dist_$child")
            push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
            if !isnothing(pgm.observed_values[child]) && !static_observes
                # recompute observe, could have changed
                push!(block_args, :($X[$child] = $(symbolic_observes[child])))
            end
            push!(block_args, :($lp += logpdf($child_d_sym, $X[$child])))   
        end
        push!(block_args, :($lp))

        f_name = Symbol("$(pgm.name)_lp_plate_$(plate.symbol)")
        plate_function_names[plate] = f_name
        f = rmlines(:(
            function $f_name($X::Vector{Float64})
                $(Expr(:block, block_args...))
            end
        ))
        # display(f)
        f = eval(f)
        push!(plate_functions, f)
    end

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
        for child in children
            if child in plates
                plate_f_name = plate_function_names[child]
                push!(block_args, :($log_α -= $plate_f_name($X)))
            else
                child_d_sym = gensym("child_dist_$child")
                push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
                if !isnothing(pgm.observed_values[child]) && !static_observes
                    # recompute observe, could have changed
                    push!(block_args, :($X[$child] = $(symbolic_observes[child])))
                end
                push!(block_args, :($log_α -= logpdf($child_d_sym, $X[$child])))
            end
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
            if child in plates
                plate_f_name = plate_function_names[child]
                push!(block_args, :($log_α += $plate_f_name($X)))
            else
                child_d_sym = gensym("child_dist_$child")
                push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
                if !isnothing(pgm.observed_values[child]) && !static_observes
                    # recompute observe, could have changed
                    push!(block_args, :($X[$child] = $(symbolic_observes[child])))
                end
                push!(block_args, :($log_α += logpdf($child_d_sym, $X[$child])))    
            end      
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
    @progress for f in plate_functions
        println(f)
        Base.invokelatest(f, X)
    end
    @progress for f in lmh_functions
        println(f)
        Base.invokelatest(f, X)
    end
    return lmh_functions
end

export lmh, compile_lmh, compiled_single_site


function compile_lmh_2(pgm::PGM; static_observes::Bool=false)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)

    lp = gensym(:lp)
    block_args = []
    push!(block_args, :($lp = 0.0))

    X = gensym(:X)
    mask = gensym(:mask)

    symbolic_dists = get_symbolic_distributions(pgm, X)
    symbolic_observes = get_symbolic_observed_values(pgm, X, static_observes)
    
    for i in pgm.topological_order
        sym = ix_to_sym[i]

        d_sym = gensym("dist_$i")
        push!(block_args, :($d_sym = $(symbolic_dists[i])))

        if haskey(pgm.symbolic_pgm.Y, sym)
            y = symbolic_observes[i]
            if static_observes
                push!(block_args, :(
                    if $mask[$i]
                        $lp += logpdf($d_sym, $y)
                    end))
            else
                push!(block_args, :($X[$i] = $y))
                push!(block_args, :(
                    if $mask[$i]
                        $lp += logpdf($d_sym, $X[$i])
                    end))
            end
        else
            push!(block_args, :(
                if $mask[$i]
                    $lp += logpdf($d_sym, $X[$i])
                end))
        end
    end

    push!(block_args, :($lp))

    f_name = Symbol("$(pgm.name)_maksed_lw")
    f = rmlines(:(
        function $f_name($X::Vector{Float64}, $mask::BitVector)
            $(Expr(:block, block_args...))
        end
    ))
    # display(f)
    lw = eval(f)
    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample(X)
    mask = trues(pgm.n_variables)
    Base.invokelatest(lw, X, mask); # compilation

    masks = BitVector[]
    for node in 1:pgm.n_variables
        for child in [child for (x,child) in pgm.edges if x == node]
            mask[child] = true
        end
        mask[node] = true
        push!(masks, mask)
    end
    return lw, masks
end

export compile_lmh_2


function compiled_lmh_2(pgm::PGM, masked_lw::Function, masks::Vector{BitVector}, n_samples::Int; static_observes::Bool=false, proposal=Proposal())
    retvals = Vector{Any}(undef, n_samples)
    trace = Array{Float64,2}(undef, pgm.n_variables, n_samples)

    observed = .!isnothing.(pgm.observed_values)

    nodes = [(n,mask) for (n,mask) in enumerate(masks) if !observed[n]]

    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample(X) # initialise
    r = pgm.return_expr(X)

    n_accepted = 0 
    @progress for i in 1:n_samples
        node, mask = rand(nodes)
        d = pgm.distributions[node](X)
        q = get(proposal, pgm.addresses[node], d)
        value_current = X[node]
        W_current = masked_lw(X, mask)

        value_proposed = rand(q)
        X[node] = value_proposed

        W_proposed = masked_lw(X, mask)
        
        log_α = W_proposed - W_current + logpdf(q, value_current) - logpdf(q, value_proposed)

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

export compiled_lmh_2