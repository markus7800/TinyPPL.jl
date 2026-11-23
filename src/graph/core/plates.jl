
struct Plate
    symbol::Symbol
    nodes::UnitRange
end
Base.show(io::IO, plate::Plate) = print(io, "Plate($(plate.symbol),$(plate.nodes))")
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
Base.show(io::IO, edge::PlateToPlateEdge) = print(io, "PlateToPlateEdge($(edge.from)->$(edge.to))")

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
Base.show(io::IO, edge::NodeToPlateEdge) = print(io, "NodeToPlateEdge($(edge.from)->$(edge.to))")

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
Base.show(io::IO, edge::PlateToNodeEdge) = print(io, "PlateToNodeEdge($(edge.from)->$(edge.to))")

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
Base.show(io::IO, edge::InterPlateEdge) = print(io, "InterPlateEdge($(edge.from)->$(edge.to))")

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
Base.show(io::IO, edge::NodeToNodeEdge) = print(io, "NodeToNodeEdge($(edge.from)->$(edge.to))")

function get_plates(n_variables::Int, edges::Set{Pair{Int,Int}}, addresses::Vector{Any}, plate_symbols::Vector{Symbol})
    # addresses of pgm are sorted by plate
    plates = Plate[]
    for plate_symbol in plate_symbols
        i = n_variables
        j = 0
        # collect all nodes that have address $plate_symbol => ...
        nodes = Vector{Int}()
        for (node, addr) in enumerate(addresses)
            if addr isa Pair && addr[1] == plate_symbol
                push!(nodes, node)
                # find start and end node
                i = min(i, node)
                j = max(j, node)
            end
        end
        # assert that these addresses are in sequence in X
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
                    delete!(edges, node=>plate_node) # processed edge
                end
                push!(plated_edges, NodeToPlateEdge(node, plate))
            end
            if all((plate_node=>node) in edges for plate_node in plate.nodes)
                # node depends on plate
                for plate_node in plate.nodes
                    delete!(edges, plate_node=>node) # processed edge
                end
                # println(plate, "->", node)
                push!(plated_edges, PlateToNodeEdge(plate, node))
            end
        end
    end
    for plate in plates, other_plate in plates
        if all(NodeToPlateEdge(plate_node, other_plate) in plated_edges for plate_node in plate.nodes)
            # other_plate depends on all nodes of plate_node
            # replace NodeToPlateEdge with PlateToPlateEdge
            for plate_node in plate.nodes
                delete!(plated_edges, NodeToPlateEdge(plate_node, other_plate))
            end
            push!(plated_edges, PlateToPlateEdge(plate, other_plate))
        end
        if all(PlateToNodeEdge(other_plate, plate_node) in plated_edges for plate_node in plate.nodes)
            # plate_node depends on all nodes of other_plate
            # replace NodeToPlateEdge with PlateToPlateEdge
            for plate_node in plate.nodes
                delete!(plated_edges, PlateToNodeEdge(other_plate, plate_node))
            end
            push!(plated_edges, PlateToPlateEdge(other_plate, plate))
        end
    end

    # if we can map each node of plate to exactly one node of other_plate
    # then we have found an InterPlateEdge
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
                delete!(edges, e) # processed edge
            end
            # if :x => i has edge to :y => i then we have found identity bijection
            is_identity = Set(x=>y for (x,y) in  zip(plate.nodes, other_plate_nodes)) == bijection
            push!(plated_edges, InterPlateEdge(plate, other_plate, bijection, is_identity))
        end            
    end
    
    for e in edges # all unprocessed edges have to be NodeToNodeEdge
        push!(plated_edges, NodeToNodeEdge(e[1], e[2]))
    end

    return plates, plated_edges
end

function get_plate_functions(plates::Vector{Plate}, plated_edges::Set{PlatedEdge}, symbolic_dists, is_observed::Vector{Bool}, X::Symbol, Y::Symbol, X_unconstrained::Symbol, kind::Symbol)
    n_latents = length(is_observed) - sum(is_observed)
    @assert kind in (:sample, :lp, :lp_unconstrained, :to_constrained, :to_unconstrained)
    static_observes = true
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

        # all nodes in plate should be either observed or not observed
        all_observed = all(is_observed[child] for child in plate.nodes)
        no_observed = all(!is_observed[child] for child in plate.nodes)
        @assert all_observed || no_observed "$plate is not either all_observed or not_observed"
        
        # println("Plate:", plate, ", static_observes: ", static_observes, ", all_observed: ", all_observed, ", no_observed: ", no_observed)

        if (kind == :sample && no_observed) || # don't "sample" observes
            (kind in (:to_constrained, :to_unconstrained) && no_observed) || # don't transform observes
            (kind in (:lp, :lp_unconstrained))

            is_iid = length(plate.nodes) > 1 && allequal([symbolic_dists[child] for child in plate.nodes])
            interplate_edges = [edge for edge in plated_edges if edge isa InterPlateEdge && edge.to == plate]
            all_identity_edges = length(interplate_edges) > 0 && all(edge.is_identity for edge in interplate_edges)

            if is_iid
                # we can compute distribution once and loop
                iid_d_sym = gensym("iid_dist")
                d = symbolic_dists[first(plate.nodes)]
                push!(block_args, :($iid_d_sym = $d))

                # transform distribution if necessary
                should_transform = no_observed && should_transform_to_unconstrained(d)
                if kind in (:lp_unconstrained, :to_constrained, :to_unconstrained) && should_transform
                    push!(block_args, :($iid_d_sym = to_unconstrained($iid_d_sym)))
                end
                
                loop_var = gensym("i")

                if kind == :sample # no_observed
                    loop_body = :($X[$loop_var] = rand($iid_d_sym))

                elseif kind == :lp || (kind == :lp_unconstrained && !should_transform) # observed or unobserved
                    array_access = no_observed ? :($X[$loop_var]) : :($Y[$loop_var - $n_latents])
                    loop_body = :($lp += logpdf($iid_d_sym, $array_access))

                elseif kind == :lp_unconstrained && should_transform # no_observed
                    loop_body = :(
                        $lp += logpdf($iid_d_sym, $X[$loop_var]); # unconstrained value
                        $X[$loop_var] = $iid_d_sym.T_inv($X[$loop_var]) # to constrained value
                    )

                elseif kind == :to_unconstrained && !should_transform # no_observed
                    loop_body = :($X_unconstrained[$loop_var] = $X[$loop_var])

                elseif kind == :to_unconstrained && should_transform # no_observed
                    loop_body = :($X_unconstrained[$loop_var] = $iid_d_sym.T($X[$loop_var]))

                elseif kind == :to_constrained && !should_transform # no_observed
                    loop_body = :($X[$loop_var] = $X_unconstrained[$loop_var])

                elseif kind == :to_constrained && should_transform # no_observed
                    loop_body = :($X[$loop_var] = $iid_d_sym.T_inv($X_unconstrained[$loop_var]))

                else
                    error("Unknown case.")
                end
                
                push!(block_args, :(
                    for $loop_var in $(plate.nodes)
                        $loop_body
                    end
                ))

            elseif all_identity_edges
                # we can express dependencies in loop body
                loop_var = gensym("i")
                plate_symbolic_dists = [deepcopy(symbolic_dists[child]) for child in plate.nodes]
                # replace X[node] accesses with X[low + i] accesses and collect symbolic dists
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
                # all dist expressions have to be the same to put them in loop body
                @assert allequal(plate_symbolic_dists) unique(plate_symbolic_dists) # TODO: make check and fallback to spaghetti otherwise
                loop_d_sym = gensym("loop_dist")
                low = first(plate.nodes)-1
                should_transform = no_observed && should_transform_to_unconstrained(plate_symbolic_dists[1])

                if kind == :sample # no_observed
                    loop_body = :($X[$low + $loop_var] = rand($loop_d_sym))

                elseif kind == :lp || (kind == :lp_unconstrained && !should_transform) # observed or unobserved
                    array_access = no_observed ? :($X[$low + $loop_var]) : :($Y[$low + $loop_var - $n_latents])
                    loop_body = :($lp += logpdf($loop_d_sym, $array_access))

                elseif kind == :lp_unconstrained && should_transform # no_observed
                    loop_body = :(
                        $loop_d_sym = to_unconstrained($loop_d_sym);
                        $lp += logpdf($loop_d_sym, $X[$low + $loop_var]); # unconstrained value
                        $X[$low + $loop_var] = $loop_d_sym.T_inv($X[$low + $loop_var]) # to constrained value
                    )

                elseif kind == :to_unconstrained && !should_transform # no_observed
                    loop_body = :($X_unconstrained[$low + $loop_var] = $X[$low + $loop_var])

                elseif kind == :to_unconstrained && should_transform # no_observed
                    loop_body = :(
                        $loop_d_sym = to_unconstrained($loop_d_sym);
                        $X_unconstrained[$low + $loop_var] = $loop_d_sym.T($X[$low + $loop_var])
                    )

                elseif kind == :to_constrained && !should_transform # no_observed
                    loop_body = :($X[$low + $loop_var] = $X_unconstrained[$low + $loop_var])

                elseif kind == :to_constrained && should_transform # no_observed
                    loop_body = :(
                        $loop_d_sym = to_unconstrained($loop_d_sym);
                        $X[$low + $loop_var] = $loop_d_sym.T_inv($X_unconstrained[$low + $loop_var])
                    )
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
                    if kind in (:lp_unconstrained, :to_constrained, :to_unconstrained) && should_transform
                        push!(block_args, :($child_d_sym = to_unconstrained($child_d_sym)))
                    end

                    if kind == :sample # no_observed
                        push!(block_args, :($X[$child] = rand($child_d_sym))) 
    
                    elseif kind == :lp || (kind == :lp_unconstrained && !should_transform) # observed or unobserved
                        array_access = no_observed ? :($X[$child]) : :($Y[$child - $n_latents])
                        push!(block_args, :($lp += logpdf($child_d_sym, $array_access)))
    
                    elseif kind == :lp_unconstrained && should_transform # no_observed
                        push!(block_args, :($lp += logpdf($child_d_sym, $X[$child])))  # unconstrained value
                        push!(block_args, :($X[$child] = $child_d_sym.T_inv($X[$child]))) # to constrained value
    
                    elseif kind == :to_unconstrained && !should_transform # no_observed
                        push!(block_args, :($X_unconstrained[$child] = $X[$child]))
    
                    elseif kind == :to_unconstrained && should_transform # no_observed
                        push!(block_args, :($X_unconstrained[$child] = $child_d_sym.T($X[$child])))
    
                    elseif kind == :to_constrained && !should_transform # no_observed
                        push!(block_args, :($X[$child] = $X_unconstrained[$child]))
    
                    elseif kind == :to_constrained && should_transform # no_observed
                        push!(block_args, :($X[$child] = $child_d_sym.T_inv($X_unconstrained[$child])))
                    else
                        error("Unknown case.")
                    end
                end
            end
        end
        if kind in (:lp, :lp_unconstrained)
            push!(block_args, :($lp))
        end
    
        if kind in (:to_constrained, :to_unconstrained)
            f = :(
                function ($X::INPUT_VECTOR_TYPE, $X_unconstrained::INPUT_VECTOR_TYPE)
                    $(block_args...)
                end
            )
        elseif kind in (:lp, :lp_unconstrained)
            f = :(
                function ($X::INPUT_VECTOR_TYPE, $Y::Vector{Float64})
                    $(block_args...)
                end
            )
        elseif kind == :sample
            f = :(
                function ($X::INPUT_VECTOR_TYPE,)
                    $(block_args...)
                end
            )
        end
        # display(f)
        push!(plate_functions, make_function(f))
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

# Returns the generated function of kind `kind` for `plate`.
# Generated code calls it by interpolating the function itself (not a global name), e.g. :($(plate_function(...))(X, Y))
function plate_function(plate_info::PlateInfo, kind::Symbol, plate::Plate)
    i = findfirst(==(plate), plate_info.plates)
    @assert !isnothing(i) "$plate not in plates"
    fs = if kind == :lp
        plate_info.plate_lp_fs
    elseif kind == :lp_unconstrained
        plate_info.plate_lp_unconstrained_fs
    elseif kind == :sample
        plate_info.plate_sample_fs
    elseif kind == :to_constrained
        plate_info.plate_to_constrained
    elseif kind == :to_unconstrained
        plate_info.plate_to_unconstrained
    else
        error("Unknown plate function kind $kind.")
    end
    return fs[i]
end

function get_topolocial_order(n_variables::Int, plate_info::PlateInfo)
    # Topological order of "units": plates and unplated nodes.
    # Plated edges may connect single plate members (NodeToNodeEdge, e.g. x[1] -> z),
    # thus each endpoint is mapped to the plate it belongs to.
    node_to_plate = Dict{Int,Plate}()
    for plate in plate_info.plates, node in plate.nodes
        node_to_plate[node] = plate
    end
    unit(x::Plate) = x
    unit(x::Int) = get(node_to_plate, x, x)

    unique_nodes = Union{Int,Plate}[]
    append!(unique_nodes, plate_info.plates) # may be plate with no edge
    append!(unique_nodes, [node for node in 1:n_variables if !haskey(node_to_plate, node)])

    unit_edges = Set{Pair{Union{Int,Plate},Union{Int,Plate}}}()
    for edge in plate_info.plated_edges
        from, to = unit(edge.from), unit(edge.to)
        from == to && continue # edges within plate are handled by plate functions
        push!(unit_edges, from => to)
    end

    n_parents = Dict{Union{Int,Plate},Int}(u => 0 for u in unique_nodes)
    for (_, to) in unit_edges
        n_parents[to] += 1
    end

    ordered_nodes = Union{Int,Plate}[] # topological order
    roots = Union{Int,Plate}[u for u in unique_nodes if n_parents[u] == 0]
    while !isempty(roots)
        u = pop!(roots)
        push!(ordered_nodes, u)
        for (from, to) in unit_edges
            if from == u
                n_parents[to] -= 1
                if n_parents[to] == 0
                    push!(roots, to)
                end
            end
        end
    end
    @assert length(ordered_nodes) == length(unique_nodes) "Cyclic dependency between plates and nodes (e.g. plate member -> node -> other plate member)."
    return ordered_nodes
end
