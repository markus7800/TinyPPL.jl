import MacroTools
import ..TinyPPL.Distributions: logpdf
using ..TinyPPL.Distributions

import Tracker
const INPUT_VECTOR_TYPE = Union{
    Vector{Float64},
    AbstractVector{Float64}, # includes Tracker.TrackedVector
    Vector{Tracker.TrackedReal{Float64}}
    #Tracker.TrackedVector{Float64,Vector{Float64}}}
}

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
    Assumption: all sample statements produce Float64 output.
=#
struct PGM
    name::Symbol
    n_variables::Int
    n_samples::Int

    edges::Set{Pair{Int,Int}} # edges
    addresses::Vector{Any}

    distributions::Vector{Function} # distributions not pdfs
    observations::Vector{Real} # observations
    return_expr::Function

    plate_info::Union{Nothing, PlateInfo}
    
    sample!::Function # stores samples (incl. observers) in input
    logpdf::Function
    unconstrained_logpdf!::Function # transforms input to constrained

    transform_to_constrained!::Function
    transform_to_unconstrained!::Function

    sym_to_ix::Dict{Symbol, Int}
    symbolic_pgm::SymbolicPGM
    symbolic_return_expr::Union{Expr, Symbol, Real}

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
macro pgm(annotations, name, foppl)
    if annotations isa Symbol
        annotations = Set{Symbol}([annotations])
    else
        annotations = Set{Symbol}(annotations.args)
    end
    return pgm_macro(annotations, name, foppl)
end

macro pgm(name, foppl)
    return pgm_macro(Set{Symbol}(), name, foppl)
end

function pgm_macro(annotations, name, foppl)
    foppl = rmlines(foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, {symbol_} ~ dist_ ↦ y_) ? Expr(:observe, symbol, dist, y) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ = dist_ ↦ y_) ? :($var = $(Expr(:observe, QuoteNode(var), dist, y))) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, dist_ ↦ y_) ? Expr(:observe, QuoteNode(:OBSERVED), dist, y) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, {symbol_} ~ dist_) ? Expr(:sample, symbol, dist) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ ~ dist_) ? :($var = $(Expr(:sample, QuoteNode(var), dist))) : expr,  foppl);
    foppl = unwrap_let(foppl)

    G, E, variable_to_address = transpile_program(foppl);
    
    # println(foppl)
    pgm = compile_symbolic_pgm(name, G, E, variable_to_address, annotations);

    return pgm
end

export @pgm


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
function subtitute_for_syms(var_to_expr::Dict{Symbol,Any}, d::Any, X::Symbol)
    d = substitute(var_to_expr, d)
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
            if !isempty(ixs)
                min_ix = minimum(ixs)
                max_ix = maximum(ixs)
                if collect(min_ix:max_ix) == ixs
                    return Expr(:ref, X, :($min_ix:$max_ix))
                end
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
function get_symbolic_distributions(symbolic_pgm::SymbolicPGM, n_samples::Int, ix_to_sym::Dict{Int,Symbol}, var_to_expr::Dict{Symbol,Any}, X::Symbol)
    symbolic_dists = []
    for node in 1:n_samples
        sym = ix_to_sym[node]
        d = symbolic_pgm.P[sym]
        d = subtitute_for_syms(var_to_expr, d, X)
        push!(symbolic_dists, d)
    end
    return symbolic_dists
end

function get_symbolic_distributions(pgm::PGM, X::Symbol)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:pgm.n_samples)
    return get_symbolic_distributions(pgm.symbolic_pgm, pgm.n_samples, ix_to_sym, var_to_expr, X)
end

# sort by address
# no nested plates, i.e (:x=>:y=>i) and (:x=>:y=>j) belong both to plate :x
function plates_lt(symbolic_pgm::SymbolicPGM, variable_to_address)
    return function lt(sym_x, sym_y)
        if !haskey(symbolic_pgm.Y, sym_x) && haskey(symbolic_pgm.Y, sym_y)
            return true
        end
        if haskey(symbolic_pgm.Y, sym_x) && !haskey(symbolic_pgm.Y, sym_y)
            return false
        end
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

const ALL_DISTRIBUTIONS = Set([
    :Bernoulli, :Binomial, :Categorical, :DiscreteUniform, :Geometric, :Poisson,
    :Beta, :Cauchy, :Exponential, :Gamma, :InverseGamma, :Laplace, :LogNormal, :Normal, :TDist, :Uniform,
    :Dirac
])
const UNCONSTRAINED_DISTRIBUTIONS = Set([
    # discrete cannot be transformed to unconstrained
    :Bernoulli, :Binomial, :Categorical, :DiscreteUniform, :Geometric, :Poisson,
    :Dirac,
    :Cauchy, :Laplace, :LogNormal, :Normal, :TDist
])
const CONSTRAINED_DISTRIBUTIONS = Set([
    :Beta, :Exponential, :Gamma, :InverseGamma, :Uniform
])

function should_transform_to_unconstrained(symbolic_dist)
    dist_names = get_call_names(symbolic_dist) ∩ ALL_DISTRIBUTIONS
    return !isempty(dist_names ∩ CONSTRAINED_DISTRIBUTIONS)
end

function get_logpdf(name::Symbol, n_variables::Int, edges::Set{Pair{Int,Int}}, plate_info::Union{Nothing,PlateInfo}, symbolic_dists, is_observed::Vector{Bool}, observations::Vector{Float64}, X::Symbol; unconstrained::Bool=false)
    n_samples = n_variables - length(observations)
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
            plate_f_name = plate_function_name(name, unconstrained ? :lp_unconstrained : :lp, node)
            push!(lp_block_args, :($lp += $plate_f_name($X)))
        else
            d = symbolic_dists[node]
            
            if is_observed[node]
                push!(lp_block_args, :($d_sym = $d))
                push!(lp_block_args, :($lp += logpdf($d_sym, $(observations[node-n_samples]))))
            else
                if unconstrained && should_transform_to_unconstrained(d)
                    push!(lp_block_args, :($d_sym = $d))
                    push!(lp_block_args, :($d_sym = to_unconstrained($d_sym)))
                    push!(lp_block_args, :($lp += logpdf($d_sym, $X[$node]))) # unconstrained value
                    push!(lp_block_args, :($X[$node] = $d_sym.T_inv($X[$node]))) # to constrained value
                else
                    push!(lp_block_args, :($d_sym = $d))
                    push!(lp_block_args, :($lp += logpdf($d_sym, $X[$node])))
                end
            end
        end
    end

    push!(lp_block_args, :($lp))

    
    f_name = !unconstrained ? Symbol("$(name)_logpdf") : Symbol("$(name)_logpdf_unconstrained!")

    f = rmlines(:(
        function $f_name($X::INPUT_VECTOR_TYPE)
            $(Expr(:block, lp_block_args...))
        end
    ))
    display(f)
    logpdf = eval(f)
    return logpdf
end


function get_transform(name::Symbol, n_variables::Int, edges::Set{Pair{Int,Int}}, plate_info::Union{Nothing,PlateInfo}, symbolic_dists, is_observed::Vector{Bool}, X::Symbol, X_unconstrained::Symbol; to::Symbol)
    # X is constrained, Y is unconstrained
    # distributions arguments are written in terms of X
    # to = :unconstrained leaves X unchanged, writes into Y
    # to = :constrained leaves Y unchanged, writes into X
    # transform_to_constrained(X, X) works
    # but transform_to_unconstrained(X, X) does not work because the constrained values that are used get overwritten
    @assert to in (:constrained, :unconstrained)
    block_args = []

    ordered_nodes = if isnothing(plate_info)
        get_topolocial_order(n_variables, edges)
    else
        get_topolocial_order(n_variables, plate_info)
    end

    d_sym = gensym("dist")
    for node in ordered_nodes
        if node isa Plate
            plate_f_name = plate_function_name(name, to == :constrained ? :to_constrained : :to_unconstrained, node)
            push!(block_args, :($plate_f_name($X,$X_unconstrained)))
        else
            d = symbolic_dists[node]
            
            if !is_observed[node]
                if should_transform_to_unconstrained(d)
                    push!(block_args, :($d_sym = $d))
                    push!(block_args, :($d_sym = to_unconstrained($d_sym)))
                    if to == :unconstrained
                        push!(block_args, :($X_unconstrained[$node] = $d_sym.T($X[$node])))
                    else # to == :constrained
                        push!(block_args, :($X[$node] = $d_sym.T_inv($X_unconstrained[$node])))
                    end
                else
                    if to == :unconstrained
                        push!(block_args, :($X_unconstrained[$node] = $X[$node]))
                    else # to == :constrained
                        push!(block_args, :($X[$node] = $X_unconstrained[$node]))
                    end
                end
            end
        end
    end

    if to == :constrained
        push!(block_args, :($X))
    else
        push!(block_args, :($X_unconstrained))
    end

    
    f_name = Symbol("$(name)_transform_to_$(to)!")

    f = rmlines(:(
        function $f_name($X::INPUT_VECTOR_TYPE, $X_unconstrained::INPUT_VECTOR_TYPE)
            $(Expr(:block, block_args...))
        end
    ))
    display(f)
    transform = eval(f)
    return transform
end

function get_sample(name::Symbol, n_variables::Int, edges::Set{Pair{Int,Int}}, plate_info::Union{Nothing,PlateInfo}, symbolic_dists, is_observed::Vector{Bool}, X::Symbol)
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
            if !is_observed[node]
                d = symbolic_dists[node]
                push!(sample_block_args, :($d_sym = $d))
                push!(sample_block_args, :($X[$node] = rand($d_sym)))
            end
        end
    end

    push!(sample_block_args, :($nothing))
    f_name = Symbol("$(name)_sample")
    f = rmlines(:(
        function $f_name($X::INPUT_VECTOR_TYPE)
            $(Expr(:block, sample_block_args...))
        end
    ))
    display(f)
    sample = eval(f)
    return sample
end

function compile_symbolic_pgm(
    name::Symbol, 
    spgm::SymbolicPGM, E::Union{Expr, Symbol, Real}, 
    variable_to_address::Dict{Symbol, Any}, 
    annotations::Set{Symbol}
    )

    n_variables = length(spgm.V)
    n_observes = length(spgm.Y)
    n_samples = n_variables - n_observes

    # we map the symbol to index in vector, sorted by address, observes last
    sym_to_ix = Dict(sym => ix for (ix, sym) in enumerate(
        sort(collect(spgm.V), lt=plates_lt(spgm, variable_to_address), alg=InsertionSort))
    )
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)
    edges = Set([sym_to_ix[x] => sym_to_ix[y] for (x, y) in spgm.A])
    # assert that observations are last
    @assert all(!isobserved(spgm, ix_to_sym[j]) for j in 1:n_samples)
    @assert all(isobserved(spgm, ix_to_sym[j]) for j in (n_samples+1):n_variables)
    is_observed = [isobserved(spgm, ix_to_sym[j]) for j in 1:n_variables]

    topolical_ordered = get_topolocial_order(n_variables, edges) # includes oberved variables

    X = gensym(:X)
    # observed variables are never read from X, and since they are static we know the observerations after compilation
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:n_samples)
    symbolic_dists = get_symbolic_distributions(spgm, n_variables, ix_to_sym, var_to_expr, X)

    distributions = Vector{Function}(undef, n_variables)
    observations = Vector{Float64}(undef, n_observes)
    addresses = Vector{Any}(undef, n_variables)

    # just wrap symbolic distributions / observed values in functions
    for i in 1:n_variables
        sym = ix_to_sym[i]
        addresses[i] = variable_to_address[sym]

        f_name = Symbol("$(name)_dist_$i")
        f = rmlines(:(
            function $f_name($X::INPUT_VECTOR_TYPE) # length($X) = n_samples
                $(symbolic_dists[i])
            end
        ))
        # display(f)
        distributions[i] = eval(f)
        if isobserved(spgm, ix_to_sym[i])
            observations[i-n_samples] = spgm.Y[sym]
        end
    end

    # wrap return expression in function
    f_name = Symbol("$(name)_return")
    new_E = subtitute_for_syms(var_to_expr, deepcopy(E), X)
    f = rmlines(:(
        function $f_name($X::INPUT_VECTOR_TYPE) # length($X) = n_samples
            $new_E
        end
    ))
    # display(f)
    return_expr = eval(f)


    X_unconstrained = gensym(:X_unconstrained) # unconstrained

    if :plated in annotations
        plate_symbols = unique([addr[1] for addr in addresses if addr isa Pair])
        println("plate_symbols: ", plate_symbols)
        plates, plated_edges = get_plates(n_variables, edges, addresses, plate_symbols)
        plate_lp_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, is_observed, X, X_unconstrained, :lp)
        plate_lp_unconstrained_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, is_observed, X, X_unconstrained, :lp_unconstrained)
        plate_sample_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, is_observed, X, X_unconstrained, :sample)
        plate_to_constrained = get_plate_functions(name, plates, plated_edges, symbolic_dists, is_observed, X, X_unconstrained, :to_constrained)
        plate_to_unconstrained = get_plate_functions(name, plates, plated_edges, symbolic_dists, is_observed, X, X_unconstrained, :to_unconstrained)
        plate_info = PlateInfo(plate_symbols, plates, plated_edges,
            plate_lp_fs,
            plate_lp_unconstrained_fs,
            plate_sample_fs,
            plate_to_constrained,
            plate_to_unconstrained
            )
    else
        plate_info = nothing
    end

    sample! = get_sample(name, n_variables, edges, plate_info, symbolic_dists, is_observed, X)
    logpdf = get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, is_observed, observations, X; unconstrained=false)
    unconstrained_logpdf! = get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, is_observed, observations, X; unconstrained=true)
    transform_to_constrained! = get_transform(name, n_variables, edges, plate_info, symbolic_dists, is_observed, X, X_unconstrained; to=:constrained)
    transform_to_unconstrained! = get_transform(name, n_variables, edges, plate_info, symbolic_dists, is_observed, X, X_unconstrained; to=:unconstrained)

    if !(:uninvoked in annotations)
        # force compilation
        X = Vector{Float64}(undef, n_samples)
        # invokes plate functions as well
        Base.invokelatest(sample!, X)
        Base.invokelatest(logpdf, X)
        Y = Vector{Float64}(undef, n_samples)
        Y = Base.invokelatest(transform_to_unconstrained!, X, Y)
        Z = Vector{Float64}(undef, n_samples)
        Base.invokelatest(transform_to_constrained!, Z, Y)
        @assert all(X .≈ Z) (X,Z)

        Y = Base.invokelatest(transform_to_unconstrained!, X, Y)
        Base.invokelatest(unconstrained_logpdf!, Y)
        @assert all(X .≈ Y) (X, X)

        Base.invokelatest(return_expr, X)
        for i in 1:n_variables
            Base.invokelatest(distributions[i], X)
        end
    end

    return PGM(
        name,
        n_variables,
        n_samples,

        edges,
        addresses,

        distributions,
        observations,
        return_expr,

        plate_info,
        sample!,
        logpdf,

        unconstrained_logpdf!,

        transform_to_constrained!,
        transform_to_unconstrained!,

        sym_to_ix,
        spgm,
        E,

        topolical_ordered
    )
end

function isobserved(pgm::PGM, node::Int)
    return node > pgm.n_samples
end
export isobserved