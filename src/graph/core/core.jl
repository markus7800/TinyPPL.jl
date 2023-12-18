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
    Assumption: all sample statements produce Float64 output.
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
    sample!::Function # stores samples (incl. observers) in input
    logpdf::Function
    unconstrained_logpdf!::Function # transforms input to constrained

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
macro ppl(annotations, name, foppl)
    if annotations isa Symbol
        annotations = Set{Symbol}([annotations])
    else
        annotations = Set{Symbol}(annotations.args)
    end
    return ppl_macro(annotations, name, foppl)
end

macro ppl(name, foppl)
    return ppl_macro(Set{Symbol}(), name, foppl)
end

function ppl_macro(annotations, name, foppl)
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
function get_symbolic_distributions(symbolic_pgm::SymbolicPGM, n_variables::Int, ix_to_sym::Dict{Int,Symbol}, var_to_expr::Dict{Symbol,Any}, X::Symbol)
    symbolic_dists = []
    for node in 1:n_variables
        sym = ix_to_sym[node]
        d = symbolic_pgm.P[sym]
        d = subtitute_for_syms(var_to_expr, d, X)
        push!(symbolic_dists, d)
    end
    return symbolic_dists
end

function get_symbolic_distributions(pgm::PGM, X::Symbol)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:pgm.n_variables)
    return get_symbolic_distributions(pgm.symbolic_pgm, pgm.n_variables, ix_to_sym, var_to_expr, X)
end

#=
    replaces symbols with corresponding vector entry X[j] for each observed value expression in symbolic_pgm.
=#
function get_symbolic_observed_values(symbolic_pgm::SymbolicPGM, n_variables::Int, ix_to_sym::Dict{Int,Symbol}, var_to_expr::Dict{Symbol,Any}, X::Symbol, static_observes::Bool)
    symbolic_observes = []
    for node in 1:n_variables
        sym = ix_to_sym[node]
        if !haskey(symbolic_pgm.Y, sym)
            push!(symbolic_observes, nothing)
        else
            y = symbolic_pgm.Y[sym]
            y = subtitute_for_syms(var_to_expr, y, X)
            if static_observes
                y = eval(y)
            end
            push!(symbolic_observes, y)
        end
    end
    return symbolic_observes
end

function get_symbolic_observed_values(pgm::PGM, X::Symbol, static_observes::Bool)
    ix_to_sym = Dict(ix => sym for (sym, ix) in pgm.sym_to_ix)
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:pgm.n_variables)
    return get_symbolic_observed_values(pgm.symbolic_pgm, pgm.n_variables, ix_to_sym, var_to_expr, X, static_observes)
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

function get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, symbolic_observes, X; unconstrained=false)
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
            
            if !isnothing(symbolic_observes[node])
                push!(lp_block_args, :($d_sym = $d))
                push!(lp_block_args, :($lp += logpdf($d_sym, $X[$node])))
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

    
    f_name = !unconstrained ? Symbol("$(name)_logpdf") : Symbol("$(name)_logpdf_unconstrained")

    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $(Expr(:block, lp_block_args...))
        end
    ))
    display(f)    
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
    spgm::SymbolicPGM, E::Union{Expr, Symbol, Real}, 
    variable_to_address::Dict{Symbol, Any}, 
    annotations::Set{Symbol}
    )

    n_variables = length(spgm.V)

    # symbol to index, sorted by address
    sym_to_ix = Dict(sym => ix for (ix, sym) in enumerate(
        sort(collect(spgm.V), lt=plates_lt(spgm, variable_to_address), alg=InsertionSort))
    )
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)
    edges = Set([sym_to_ix[x] => sym_to_ix[y] for (x, y) in spgm.A])

    topolical_ordered = get_topolocial_order(n_variables, edges)

    X = gensym(:X)
    var_to_expr = Dict{Symbol,Any}(ix_to_sym[j] => :($X[$j]) for j in 1:n_variables)
    symbolic_dists = get_symbolic_distributions(spgm, n_variables, ix_to_sym, var_to_expr, X)
    static_observes = :plated in annotations
    symbolic_observes = get_symbolic_observed_values(spgm, n_variables, ix_to_sym, var_to_expr, X, static_observes)

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
    new_E = subtitute_for_syms(var_to_expr, deepcopy(E), X)
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $new_E
        end
    ))
    # display(f)
    return_expr = eval(f)

    if :plated in annotations
        plate_symbols = unique([addr[1] for addr in addresses if addr isa Pair])
        println("plate_symbols: ", plate_symbols, ", static_observes: ", static_observes)
        plates, plated_edges = get_plates(n_variables, edges, addresses, plate_symbols)
        plate_lp_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, symbolic_observes, X, static_observes, :lp)
        plate_lp_unconstrained_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, symbolic_observes, X, static_observes, :lp_unconstrained)
        plate_sample_fs = get_plate_functions(name, plates, plated_edges, symbolic_dists, symbolic_observes, X, static_observes, :sample)
        plate_info = PlateInfo(plate_symbols, plates, plated_edges, plate_lp_fs, plate_lp_unconstrained_fs, plate_sample_fs)
    else
        plate_info = nothing
    end

    sample! = get_sample(name, n_variables, edges, plate_info, symbolic_dists, symbolic_observes, X)
    logpdf = get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, symbolic_observes, X; unconstrained=false)
    unconstrained_logpdf! = get_logpdf(name, n_variables, edges, plate_info, symbolic_dists, symbolic_observes, X; unconstrained=true)

    if !(:uninvoked in annotations)
        # force compilation
        X = Vector{Float64}(undef, n_variables)
        # invokes plate functions as well
        Base.invokelatest(sample!, X)
        Base.invokelatest(logpdf, X)
        # Base.invokelatest(unconstrained_logpdf!, X) TODO: how to do this?
        Base.invokelatest(return_expr, X)
        for i in 1:n_variables
            Base.invokelatest(distributions[i], X)
            if !isnothing(observed_values[i])
                Base.invokelatest(observed_values[i], X)
            end
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
        sample!,
        logpdf,
        unconstrained_logpdf!,

        sym_to_ix,
        spgm,
        E,

        topolical_ordered
    )
end