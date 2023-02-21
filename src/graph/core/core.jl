import MacroTools
import ..TinyPPL.Distributions: logpdf
using ..TinyPPL.Distributions

include("metaprogramming_utils.jl")
include("symbolic_pgm.jl")
include("plates.jl")

#=
    Concrete PGM for sympbolic PGM.
    Generated sample / observe symbols are mapped to index in Float64 vector X.
    Symbolic expressions are compiled to functions with input X.
    E.g. for `sym=>i ` in `sym_to_ix`, `distributions[i](X)` returns the distribution
    of random variable with symbol `sym` for assigments `X`.
    You can start with an uninitialised X, and sample and assign
    `X[i] = rand(distributions[i](X))`
    in topological order.
    In fact, this is exactly what the `sample` function does.
=#
struct PGM
    name::Symbol
    n_variables::Int
    edges::Set{Pair{Int,Int}} # edges
    addresses::Vector{Any}

    distributions::Vector{Function} # distributions not pdfs
    observed_values::Vector{Union{Nothing,Function}} # observations
    return_expr::Function

    plate_info::Union{Nothing, PlateInfo}
    sample::Function
    logpdf::Function

    sym_to_ix::Dict{Symbol, Int}
    symbolic_pgm::SymbolicPGM
    symbolic_return_expr::Union{Expr, Symbol}

    topological_order::Vector{Int}
end

include("pretty_print.jl")

#=
    replaces
    `{addr} ~ distribution` with `Expr(:sample, addr, distribution)`
    `var = distributio`n with `var = Expr(:sample, var, distribution)`
    `{addr} ~ distribution ↦ y` with `Expr(:observe, addr, distribution, y)`
    `var = distribution ↦ y` with `var = Expr(:observe, var, distribution, y)`
    unwraps let expressions
    and transpiles to symbolic PGM, which is compiled to concrete PGM.
=#
macro ppl(annotation, name, foppl)
    return ppl_macro(annotation, name, foppl)
end

macro ppl(name, foppl)
    return ppl_macro(nothing, name, foppl)
end

function ppl_macro(annotation, name, foppl)
    foppl = rmlines(foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, {symbol_} ~ dist_ ↦ y_) ? Expr(:observe, symbol, dist, y) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ = dist_ ↦ y_) ? :($var = $(Expr(:observe, QuoteNode(var), dist, y))) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, dist_ ↦ y_) ? Expr(:observe, QuoteNode(:OBSERVED), dist, y) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, {symbol_} ~ dist_) ? Expr(:sample, symbol, dist) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ ~ dist_) ? :($var = $(Expr(:sample, QuoteNode(var), dist))) : expr,  foppl);
    foppl = unwrap_let(foppl)
    
    G, E, variable_to_address = transpile_program(foppl);
    
    pgm = compile_symbolic_pgm(name, G, E, variable_to_address, annotation);

    return pgm
end

export @ppl


function get_topolocial_order(n_variables::Int, edges::Set{Pair{Int,Int}})
    edges = deepcopy(edges)
    roots = [i for i in 1:n_variables if !any(i == y for (x,y) in edges)]
    ordered_nodes = Int[] # topological order
    nodes = Set(roots)
    while !isempty(nodes)
        node = pop!(nodes)
        push!(ordered_nodes, node)
        children = [y for (x,y) in edges if x == node]
        for child in children
            delete!(edges, node=>child)
            parents = [x for (x,y) in edges if y == child]
            if isempty(parents)
                push!(nodes, child)
            end
        end
    end
    @assert length(ordered_nodes) == n_variables
    return ordered_nodes
end

#=
    for `j=>sym` in `ix_to_sym` replaces sym with `X[j]`
=#
function subtitute_for_syms(n_variables::Int, ix_to_sym::Dict{Int,Symbol}, d::Any, X::Symbol)
    for j in 1:n_variables
        d = substitute(ix_to_sym[j], :($X[$j]), d)
    end
    d = reduce_array_accesses(d, X)
    d = unnest_array_accesses(d, X)
    return d
end

#=
    replaces [X[i], X[i+1], ... X[j]] with X[i:j]
=#
function reduce_array_accesses(expr, X::Symbol)
    if expr isa Expr
        if expr.head == :vect
            ixs = []
            for el in expr.args
                if el isa Expr && el.head == :ref && el.args[1] == X && el.args[2] isa Int
                    push!(ixs, el.args[2])
                end
            end
            min_ix = minimum(ixs)
            max_ix = maximum(ixs)
            if collect(min_ix:max_ix) == ixs
                return Expr(:ref, X, :($min_ix:$max_ix))
            end
        end
        return Expr(expr.head, [reduce_array_accesses(arg, X) for arg in expr.args]...)
    else
        return expr
    end
end

#=
    replaces X[i:j][k] with X[i-1+k]
=#
function unnest_array_accesses(expr, X::Symbol)
    if expr isa Expr
        if expr.head == :ref
            arr = expr.args[1]
            if arr isa Expr && arr.head == :ref && arr.args[1] == X
                range = arr.args[2]
                if range isa Expr && range.head == :call && range.args[1] == :(:)
                    low = range.args[2]-1
                    return Expr(:ref, X, :($low + $(expr.args[2])))
                end
            end
        end
        return Expr(expr.head, [unnest_array_accesses(arg, X) for arg in expr.args]...)
    else
        return expr

    end
end

#=
    replaces symbols with corresponding vector entry X[j] for each distribution expression in symbolic_pgm.
=#
function get_symbolic_distributions(symbolic_pgm::SymbolicPGM, n_variables::Int, sym_to_ix::Dict{Symbol,Int}, X::Symbol)
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)
    symbolic_dists = []
    for node in 1:n_variables
        sym = ix_to_sym[node]
        d = symbolic_pgm.P[sym]
        d = subtitute_for_syms(n_variables, ix_to_sym, d, X)
        push!(symbolic_dists, d)
    end
    return symbolic_dists
end

function get_symbolic_distributions(pgm::PGM, X::Symbol)
    return get_symbolic_distributions(pgm.symbolic_pgm, pgm.n_variables, pgm.sym_to_ix, X)
end

#=
    replaces symbols with corresponding vector entry X[j] for each observed value expression in symbolic_pgm.
=#
function get_symbolic_observed_values(symbolic_pgm::SymbolicPGM, n_variables::Int, sym_to_ix::Dict{Symbol,Int}, X::Symbol, static_observes::Bool)
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)

    symbolic_observes = []
    for node in 1:n_variables
        sym = ix_to_sym[node]
        if !haskey(symbolic_pgm.Y, sym)
            push!(symbolic_observes, nothing)
        else
            y = symbolic_pgm.Y[sym]
            y = subtitute_for_syms(n_variables, ix_to_sym, y, X)
            if static_observes
                y = eval(y)
            end
            push!(symbolic_observes, y)
        end
    end
    return symbolic_observes
end

function get_symbolic_observed_values(pgm::PGM, X::Symbol, static_observes::Bool)
    return get_symbolic_observed_values(pgm.symbolic_pgm, pgm.n_variables, pgm.sym_to_ix, X, static_observes)
end

# sort by address
# no nested plates, i.e (:x=>:y=>i) and (:x=>:y=>j) belong both to plate :x
function plates_lt(variable_to_address)
    return function lt(sym_x, sym_y)
        x = variable_to_address[sym_x]
        y = variable_to_address[sym_y]
        if x isa Pair && y isa Pair
            if x[1] == y[1]
                return x[2] < y[2]
            else
                return x[1] < y[1]
            end
        elseif !(x isa Pair) && !(y isa Pair)
            return x < y
        else
            return y isa Pair
        end
    end
end

function get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, X)
    lp = gensym(:lp)
    lp_block_args = []

    ordered_nodes = if isnothing(plate_info)
        get_topolocial_order(n_variables, edges)
    else
        get_topolocial_order(n_variables, plate_info)
    end

    push!(lp_block_args, :($lp = 0.0))
    d_sym = gensym("dist")
    for node in ordered_nodes
        if node isa Plate
            plate_f_name = plate_function_name(name, :lp, node)
            push!(lp_block_args, :($lp += $plate_f_name($X)))
        else
            d = symbolic_dists[node]
            push!(lp_block_args, :($d_sym = $d))
            push!(lp_block_args, :($lp += logpdf($d_sym, $X[$node])))
        end
    end

    push!(lp_block_args, :($lp))

    f_name = Symbol("$(name)_logpdf")
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $(Expr(:block, lp_block_args...))
        end
    ))
    # display(f)
    logpdf = eval(f)
    return logpdf
end

function get_sample(name, n_variables, edges, plate_info, symbolic_dists, symbolic_observes, X)
    sample_block_args = []

    ordered_nodes = if isnothing(plate_info)
        get_topolocial_order(n_variables, edges)
    else
        get_topolocial_order(n_variables, plate_info)
    end

    d_sym = gensym("dist")
    for node in ordered_nodes
        
        if node isa Plate
            plate_f_name = plate_function_name(name, :sample, node)
            push!(sample_block_args, :($plate_f_name($X)))
        else
            if !isnothing(symbolic_observes[node])
                y = symbolic_observes[node]
                push!(sample_block_args, :($X[$node] = $y))
            else
                d = symbolic_dists[node]
                push!(sample_block_args, :($d_sym = $d))
                push!(sample_block_args, :($X[$node] = rand($d_sym)))
            end
        end
    end

    push!(sample_block_args, :($nothing))
    f_name = Symbol("$(name)_sample")
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $(Expr(:block, sample_block_args...))
        end
    ))
    # display(f)
    sample = eval(f)
    return sample
end

function compile_symbolic_pgm(
    name::Symbol, 
    spgm::SymbolicPGM, E::Union{Expr, Symbol}, 
    variable_to_address::Dict{Symbol, Any}, 
    annotation::Union{Nothing, Symbol}
    )

    n_variables = length(spgm.V)

    # symbol to index, sorted by address
    sym_to_ix = Dict(sym => ix for (ix, sym) in enumerate(
        sort(collect(spgm.V), lt=plates_lt(variable_to_address), alg=InsertionSort))
    )
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)
    edges = Set([sym_to_ix[x] => sym_to_ix[y] for (x, y) in spgm.A])

    topolical_ordered = get_topolocial_order(n_variables, edges)

    X = gensym(:X)
    symbolic_dists = get_symbolic_distributions(spgm, n_variables, sym_to_ix, X)
    static_observes = annotation == :plated
    symbolic_observes = get_symbolic_observed_values(spgm, n_variables, sym_to_ix, X, static_observes)

    distributions = Vector{Function}(undef, n_variables)
    observed_values = Vector{Union{Nothing,Function}}(undef, n_variables)
    addresses = Vector{Any}(undef, n_variables)

    # just wrap symbolic distributions / observed values in functions
    for i in 1:n_variables
        sym = ix_to_sym[i]
        addresses[i] = variable_to_address[sym]

        f_name = Symbol("$(name)_dist_$i")
        f = rmlines(:(
            function $f_name($X::AbstractVector{Float64})
                $(symbolic_dists[i])
            end
        ))
        # display(f)
        distributions[i] = eval(f)
        if isnothing(symbolic_observes[i])
            observed_values[i] = nothing
        else
            f_name = Symbol("$(name)_obs_$i")
            f = rmlines(:(
                function $f_name($X::AbstractVector{Float64})
                    $(symbolic_observes[i])
                end
            ))
            # display(f)
            observed_values[i] = eval(f)
        end
    end

    # wrap return expression in function
    f_name = Symbol("$(name)_return")
    new_E =  subtitute_for_syms(n_variables, ix_to_sym, deepcopy(E), X)
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $new_E
        end
    ))
    # display(f)
    return_expr = eval(f)

    if annotation == :plated
        plate_symbols = unique([addr[1] for addr in addresses if addr isa Pair])
        println("plate_symbols: ", plate_symbols)
        plates, plated_edges = get_plates(n_variables, edges, addresses, plate_symbols)
        plate_lp_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, symbolic_observes, X, static_observes, :lp)
        plate_sample_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, symbolic_observes, X, static_observes, :sample)
        plate_info = PlateInfo(plate_symbols, plates, plated_edges, plate_lp_fs, plate_sample_fs)
    else
        plate_info = nothing
    end

    sample = get_sample(name, n_variables, edges, plate_info, symbolic_dists, symbolic_observes, X)
    logpdf = get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, X)

    # force compilation
    X = Vector{Float64}(undef, n_variables)
    Base.invokelatest(sample, X)
    Base.invokelatest(return_expr, X)
    Base.invokelatest(logpdf, X)
    for i in 1:n_variables
        Base.invokelatest(distributions[i], X)
        if !isnothing(observed_values[i])
            Base.invokelatest(observed_values[i], X)
        end
    end

    return PGM(
        name,
        n_variables,
        edges,
        addresses,

        distributions,
        observed_values,
        return_expr,

        plate_info,
        sample,
        logpdf,

        sym_to_ix,
        spgm,
        E,

        topolical_ordered
    )
end