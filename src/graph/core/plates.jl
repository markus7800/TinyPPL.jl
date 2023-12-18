
struct Plate
    symbol::Symbol
    nodes::UnitRange
end

function Base.isequal(l::Plate, r::Plate)
    return l.symbol == r.symbol
end
function Base.hash(plate::Plate)
    return hash(plate.symbol)
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
    is_identity::Bool
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

function get_plates(n_variables::Int, edges::Set{Pair{Int,Int}}, addresses::Vector{Any}, plate_symbols::Vector{Symbol})
    # addresses of pgm are sorted by plate
    plates = Plate[]
    for plate_symbol in plate_symbols
        i = n_variables
        j = 0
        nodes = Vector{Int}()
        for (node, addr) in enumerate(addresses)
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
    edges = deepcopy(edges)

    for node in 1:n_variables
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
            is_identity = Set(x=>y for (x,y) in  zip(plate.nodes, other_plate_nodes)) == bijection
            push!(plated_edges, InterPlateEdge(plate, other_plate, bijection, is_identity))
        end            
    end
    
    for e in edges
        push!(plated_edges, NodeToNodeEdge(e[1], e[2]))
    end

    return plates, plated_edges
end

function plate_function_name(name::Symbol, kind::Symbol, plate::Plate)
    Symbol("$(name)_$(kind)_plate_$(plate.symbol)")
end

function get_plate_functions(pgm_name, plates, plated_edges, symbolic_dists, symbolic_observes, X::Symbol, Y::Symbol, static_observes::Bool, kind)
    @assert kind in (:sample, :lp, :lp_unconstrained, :to_constrained, :to_unconstrained)
    # X is constrained, Y is unconstrained
    # distributions arguments are written in terms of X
    # to = :unconstrained leaves X unchanged, writes into Y
    # to = :constrained leaves Y unchanged, writes into X

    plate_functions = Function[]
    for plate in plates
        block_args = []
        lp = gensym(:lp)
        
        if kind in (:lp, :lp_unconstrained)
            push!(block_args, :($lp = 0.0))
        end

        all_observed = all(!isnothing(symbolic_observes[child]) for child in plate.nodes)
        no_observed = all(isnothing(symbolic_observes[child]) for child in plate.nodes)
        @assert all_observed || no_observed
        
        # println("Plate:", plate, ", static_observes: ", static_observes, ", all_observed: ", all_observed, ", no_observed: ", no_observed)

        if !static_observes || kind == :sample
            for child in plate.nodes
                if !isnothing(symbolic_observes[child])
                    # recompute observe, could have changed
                    push!(block_args, :($X[$child] = $(symbolic_observes[child])))
                end
            end
        end
        if kind != :sample || (kind == :sample && no_observed) # don't "sample" observes
            is_iid = length(plate.nodes) > 1 && allequal([symbolic_dists[child] for child in plate.nodes])
            interplate_edges = [edge for edge in plated_edges if edge isa InterPlateEdge && edge.to == plate]
            all_identity_edges = length(interplate_edges) > 0 && all(edge.is_identity for edge in interplate_edges)

            if is_iid
                # we can compute distribution once and loop
                iid_d_sym = gensym("iid_dist")
                d = symbolic_dists[first(plate.nodes)]
                push!(block_args, :($iid_d_sym = $d))
                should_transform = no_observed && should_transform_to_unconstrained(d)
                if kind in (:lp_unconstrained, :to_constrained, :to_unconstrained) && should_transform
                    push!(block_args, :($iid_d_sym = to_unconstrained($iid_d_sym)))
                end
                
                loop_var = gensym("i")
                loop_body = if kind == :lp_unconstrained && should_transform
                    :(begin
                        $lp += logpdf($iid_d_sym, $X[$loop_var]) # unconstrained value
                        $X[$loop_var] = $iid_d_sym.T_inv($X[$loop_var]) # to constrained value
                    end)
                elseif kind == :lp || (kind == :lp_unconstrained && !should_transform)
                    :($lp += logpdf($iid_d_sym, $X[$loop_var]))
                elseif kind == :to_constrained && should_transform
                    :($X[$loop_var] = $iid_d_sym.T_inv($Y[$loop_var]))
                elseif kind == :to_unconstrained && should_transform
                    :($Y[$loop_var] = $iid_d_sym.T($X[$loop_var]))
                elseif kind == :to_constrained !should_transform
                    :($X[$loop_var] = $Y[$loop_var])
                elseif kind == :to_unconstrained !should_transform
                    :($Y[$loop_var] = $X[$loop_var])
                elseif kind == :sample
                    :($X[$loop_var] = rand($iid_d_sym))
                else
                    error("Unknown case.")
                end
                
                push!(block_args, :(
                    for $loop_var in $(plate.nodes)
                        $loop_body
                    end
                ))

            elseif all_identity_edges
                # we can express dependencies in loop
                loop_var = gensym("i")
                plate_symbolic_dists = [deepcopy(symbolic_dists[child]) for child in plate.nodes]
                for edge in interplate_edges
                    from_low = first(edge.from.nodes)-1
                    to_low = first(edge.to.nodes)-1
                    for (i, d) in enumerate(plate_symbolic_dists)
                        for node in edge.from.nodes
                            d = substitute_expr(:($X[$node]), :($X[$(from_low) + $loop_var]), d)
                        end
                        for node in edge.to.nodes
                            d = substitute_expr(:($X[$node]), :($X[$(to_low) + $loop_var]), d)
                        end
                        plate_symbolic_dists[i] = d
                    end
                end
                @assert allequal(plate_symbolic_dists) unique(plate_symbolic_dists)
                loop_d_sym = gensym("loop_dist")
                low = first(plate.nodes)-1
                should_transform = no_observed && should_transform_to_unconstrained(plate_symbolic_dists[1])

                loop_body = if kind == :lp_unconstrained && should_transform
                    :(begin
                        $loop_d_sym = to_unconstrained($loop_d_sym)
                        $lp += logpdf($loop_d_sym, $X[$low + $loop_var]) # unconstrained value
                        $X[$low + $loop_var] = $loop_d_sym.T_inv($X[$low + $loop_var]) # to constrained value
                    end)
                elseif kind == :lp || kind == :lp_unconstrained
                    :($lp += logpdf($loop_d_sym, $X[$low + $loop_var]))
                elseif kind == :to_constrained && should_transform
                    :(begin
                        $loop_d_sym = to_unconstrained($loop_d_sym)
                        $X[$low + $loop_var] = $loop_d_sym.T_inv($Y[$low + $loop_var])
                    end)
                elseif kind == :to_unconstrained && should_transform
                    :(begin
                        $loop_d_sym = to_unconstrained($loop_d_sym)
                        $Y[$low + $loop_var] = $loop_d_sym.T($X[$low + $loop_var])
                    end)
                elseif kind == :to_constrained && !should_transform
                    :($X[$low + $loop_var] = $Y[$low + $loop_var])
                elseif kind == :to_unconstrained && !should_transform
                    :($Y[$low + $loop_var] = $X[$low + $loop_var])
                elseif kind == :sample
                    :($X[$low + $loop_var] = rand($loop_d_sym))
                else
                    error("Unknown case.")
                end
                push!(block_args, :(
                    for $loop_var in 1:$(length(plate.nodes))
                        $loop_d_sym = $(plate_symbolic_dists[1])
                        $loop_body
                    end
                ))
            else
                # cannot do any optimization => fall back to spaghetti code
                for child in plate.nodes
                    child_d_sym = gensym("child_dist_$child")
                    push!(block_args, :($child_d_sym = $(symbolic_dists[child])))
                    should_transform = no_observed && should_transform_to_unconstrained(symbolic_dists[child])
                    if should_transform
                        push!(block_args, :($child_d_sym = to_unconstrained($child_d_sym)))
                    end
                    if kind == :lp_unconstrained && should_transform
                        push!(block_args, :($lp += logpdf($child_d_sym, $X[$child])))  # unconstrained value
                        push!(block_args, :($X[$child] = $child_d_sym.T_inv($X[$child]))) # to constrained value
                    elseif kind == :lp || kind == :lp_unconstrained
                        push!(block_args, :($lp += logpdf($child_d_sym, $X[$child])))
                    elseif kind == :to_constrained && should_transform
                        push!(block_args, :($X[$child] = $child_d_sym.T_inv($Y[$child])))
                    elseif kind == :to_unconstrained && should_transform
                        push!(block_args, :($Y[$child] = $child_d_sym.T($X[$child])))
                    elseif kind == :to_constrained &&!should_transform
                        push!(block_args, :($X[$child] = $Y[$child]))
                    elseif kind == :to_unconstrained &&!should_transform
                        push!(block_args, :($Y[$child] = $X[$child]))
                    elseif kind == :sample
                        push!(block_args, :($X[$child] = rand($child_d_sym)))   
                    else
                        error("Unknown case.")
                    end
                end
            end
        end
        if kind in (:lp, :lp_unconstrained)
            push!(block_args, :($lp))
        end
    
        f_name = plate_function_name(pgm_name, kind, plate)
        if kind in (:to_constrained, :to_unconstrained)
            f = rmlines(:(
                function $f_name($X::Vector{Float64}, $Y::Vector{Float64})
                    $(Expr(:block, block_args...))
                end
            ))
            # display(f)
        else
            f = rmlines(:(
                function $f_name($X::Vector{Float64})
                    $(Expr(:block, block_args...))
                end
            ))
        end
        # display(f)
        f = eval(f)
        push!(plate_functions, f)
    end
    return plate_functions
end

struct PlateInfo
    plate_symbols::Vector{Symbol}
    plates::Vector{Plate}
    plated_edges::Set{PlatedEdge}
    plate_lp_fs::Vector{Function}
    plate_lp_unconstrained_fs::Vector{Function}
    plate_sample_fs::Vector{Function}
    plate_to_constrained::Vector{Function}
    plate_to_unconstrained::Vector{Function}
end

function get_topolocial_order(n_variables::Int, plate_info::PlateInfo)
    edges = deepcopy(plate_info.plated_edges)
    unique_nodes = Any[]
    unplated_nodes = Set(collect(1:n_variables))
    for plate in plate_info.plates
        for node in plate.nodes
            delete!(unplated_nodes, node)
        end
        push!(unique_nodes, plate) # may be plate with no edge
    end
    append!(unique_nodes, unplated_nodes)

    roots = [i for i in unique_nodes if !any(edge.to == i for edge in edges)]
    ordered_nodes = Union{Int,Plate}[] # topological order
    nodes = Set{Union{Int,Plate}}(roots)
    while !isempty(nodes)
        node = pop!(nodes)
        push!(ordered_nodes, node)
        children_edges = [edge for edge in edges if edge.from == node]
        for edge in children_edges
            child = edge.to
            delete!(edges, edge)
            parents = [edge.from for edge in edges if edge.to == child]
            if isempty(parents)
                push!(nodes, child)
            end
        end
    end
    @assert length(ordered_nodes) == length(plate_info.plates) + length(unplated_nodes)
    return ordered_nodes
end
