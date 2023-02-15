import MacroTools
import ..TinyPPL.Distributions: logpdf
using ..TinyPPL.Distributions

function unwrap_let(expr)
    if expr isa Expr
        if expr.head == :let
            @assert length(expr.args) == 2
            bindings = expr.args[1]
            @assert bindings.head == :block || bindings.head == :(=) bindings
            body = unwrap_let(expr.args[2])
            @assert body.head == :block body.head

            if bindings.head == :(=)
                return Expr(expr.head, unwrap_let.(expr.args)...)
            end
            if bindings.head == :block
                @assert all(arg.head == :(=) for arg in bindings.args) bindings.args
                # @assert length(bindings.args) > 1 bindings.args, single assignments can alos be in block

                # println("body: ", body)
                current_let = Expr(:let, unwrap_let(bindings.args[end]), body)
                for binding in reverse(bindings.args[1:end-1])
                    # println(binding)
                    current_let = Expr(:let, unwrap_let(binding), current_let)
                end
                return current_let
            end
        else
            return Expr(expr.head, unwrap_let.(expr.args)...)
        end
    else
        return expr
    end
end

function get_let_variables(expr, vars=Symbol[])
    if expr isa Expr
        if expr.head == :let
            bindings = expr.args[1]
            @assert bindings.head == :(=)
            push!(vars, bindings.args[1])
            get_let_variables(bindings.args[2], vars)
            get_let_variables(expr.args[2], vars)
        else
            for arg in expr.args
                get_let_variables(arg, vars)
            end
        end
    end
    return vars
end

function get_free_variables(expr, free=Set{Symbol}(), bound=Symbol[])
    if expr isa Symbol
        if !(expr in bound)
            push!(free, expr)
        end
    elseif expr isa Expr
        if expr.head == :let 
            binding = expr.args[1]
            variable = binding.args[1]
            body = expr.args[2]
            @assert binding.head == :(=)
            get_free_variables(binding.args[2], free, bound)
            push!(bound, variable)
            get_free_variables(body, free, bound)
            @assert pop!(bound) == variable
        else
            for arg in expr.args
                get_free_variables(arg, free, bound)
            end
        end
    end
    return free
end


function substitute(var::Symbol, with, in_expr)
    if in_expr isa Expr
        if in_expr.head == :let 
            binding = in_expr.args[1]
            @assert binding.head == :(=)
            if binding.args[1] == var
                # redefinition of variable, do not substitute
                # however, old variable can still be in expression that we assign
                return Expr(:let,
                    Expr(:(=), var, substitute(var, with, binding.args[2])),
                in_expr.args[2])
            end
        end
        return Expr(in_expr.head, [substitute(var, with, arg) for arg in in_expr.args]...)
    elseif in_expr isa Symbol && in_expr == var
        return with
    else
        return in_expr
    end
end

function simplify_if(expr)
    if expr isa Expr
        if expr.head == :if
            if expr.args[1] == true
                return expr.args[2]
            end
        end
        if expr.head == :&&
            if expr.args[1] == true
                return expr.args[2]
            end
        end
        return Expr(expr.head, simplify_if.(expr.args)...)
    else
        return expr
    end
end


# Computes the Probabilistic Graphical Model of a FOTinyPPL program.

struct SymbolicPGM
    V::Set{Symbol} # vertices
    A::Set{Pair{Symbol,Symbol}} # edges
    P::Dict{Symbol, Any} # distributions not pdfs
    Y::Dict{Symbol, Any} # observations
end

function Base.show(io::IO, spgm::SymbolicPGM)
    println(io, "Symbolic PGM")
    println(io, "Variables:")
    for v in sort(collect(spgm.V))
        println(io, v, " ~ ", spgm.P[v])
    end
    println(io, "Observed:")
    for v in sort(collect(keys(spgm.Y)))
        println(io, v, " = ", spgm.Y[v])
    end
    print(io, "Dependencies:")
    for (x,y) in sort(collect(spgm.A))
        print(io, "\n", x, " → ", y)
    end
end

function EmptyPGM()
    return SymbolicPGM(
        Set{Symbol}(),
        Set{Pair{Symbol,Symbol}}(),
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(),
    )
end

function graph_disjoint_union(G1::SymbolicPGM, G2::SymbolicPGM)
    v1 = G1.V
    v2 = G2.V
    @assert length(v1 ∩ v2) == 0 (v1, v2)
    p1 = Set(k for (k, v) in G1.P)
    p2 = Set(k for (k, v) in G2.P)
    @assert length(p1 ∩ p2) == 0 (p1, p2)
    y1 = Set(k for (k, v) in G1.Y)
    y2 = Set(k for (k, v) in G2.Y)
    @assert length(y1 ∩ y2) == 0 (y1, y2)

    return SymbolicPGM(G1.V ∪ G2.V, G1.A ∪ G2.A, Dict(G1.P ∪ G2.P), Dict(G1.Y ∪ G2.Y))
end

struct PGMTranspiler
    procs::Dict{Symbol, Expr}
    all_let_variables::Set{Symbol}
    variables::Set{Symbol}
    variable_to_address::Dict{Symbol, Any}
    function PGMTranspiler()
        return new(Dict{Symbol, Expr}(), Set{Symbol}(), Set{Symbol}(), Dict{Symbol, Any}())
    end
end

function transpile(t::PGMTranspiler, phi, expr::Symbol)
    return EmptyPGM(), expr
end

function transpile(t::PGMTranspiler, phi, expr::Real)
    return EmptyPGM(), expr
end

function transpile(t::PGMTranspiler, phi, expr::String)
    return EmptyPGM(), expr
end

function transpile(t::PGMTranspiler, phi, expr::Expr)
    if expr.head == :block
        # @assert length(expr.args) == 1 expr.args
        G = EmptyPGM()
        E = nothing
        for arg in expr.args
            g, E = transpile(t, phi, arg)
            G = graph_disjoint_union(G, g)
        end
        return G, E
    elseif expr.head == :let
        #=
        let variable = binding
            body
        end
        =#
        @assert length(expr.args) == 2
        let_head = expr.args[1]
        @assert let_head.head == :(=)
        @assert let_head.args[1] isa Symbol
        variable = let_head.args[1]
        binding = let_head.args[2]
        body = expr.args[2]
        G1, E1 = transpile(t, phi, binding) # e1 == exp.binding
        e2_sub = substitute(variable, E1, body) # e2 == exp.body
        G2, E2 = transpile(t, phi, e2_sub)
        G = graph_disjoint_union(G1, G2)
        return G, E2

    elseif expr.head == :if
        #=
        if condition
            holds
        else
            otherwise
        end
        =#
        @assert length(expr.args) == 3 # has to be if else
        condition = expr.args[1]
        holds = expr.args[2]
        otherwise = expr.args[3]
        G1, E1 = transpile(t, phi, condition)
        G2, E2 = transpile(t, :($phi && $E1), holds)
        G3, E3 = transpile(t, :($phi && !($E1)), otherwise)
        G = graph_disjoint_union(G1, G2)
        G = graph_disjoint_union(G, G3)
        E = :($E1 ? $E2 : $E3)
        return G, E

    elseif expr.head == :call
        name = expr.args[1]
        @assert name isa Symbol (name, typeof(name))
        arguments = expr.args[2:end]
        res = [transpile(t, phi, arg) for arg in arguments]
        Gs = [r[1] for r in res]
        Es = [r[2] for r in res]

        G = EmptyPGM()
        for Gi in Gs
            G = graph_disjoint_union(G, Gi)
        end

        Gh, Eh = transpile(t, phi, name) # is redundant as we only allow symbols as name
        G = graph_disjoint_union(G, Gh)
        @assert Eh isa Symbol

        if Eh in keys(t.procs)
            f = t.procs[Eh]
            f_def = f.args[1]
            body = f.args[2]
            arguments = f_def.args[2:end]
            @assert length(arguments) == length(Es)
            for (arg, Ei) in zip(arguments, Es)
                body = substitute(arg, Ei, body)
            end
            G_proc, E = transpile(t, phi, body)
            G = graph_disjoint_union(G, G_proc)
            return G_proc, E
        end

        E = Expr(:call, Eh, Es...)
        return G, E

    elseif expr.head == :vect
        G = EmptyPGM()
        E = []
        for v in expr.args
            Gv, Ev = transpile(t, phi, v)
            G = graph_disjoint_union(G, Gv)
            push!(E, Ev)
        end
        return G, Expr(:vect, E...)

    elseif expr.head == :tuple
        G = EmptyPGM()
        E = []
        for v in expr.args
            Gv, Ev = transpile(t, phi, v)
            G = graph_disjoint_union(G, Gv)
            push!(E, Ev)
        end
        return G, Expr(:tuple, E...)

    elseif expr.head == :ref
        @assert length(expr.args) == 2

        G_arr, arr = transpile(t, phi, expr.args[1])
        if expr.args[2] isa Int && arr.head == :vect
            return G_arr, arr.args[expr.args[2]]
        end
        G_ix, ix = transpile(t, phi, expr.args[2])
        G = graph_disjoint_union(G_arr, G_ix)
        E = Expr(:ref, arr, ix)
        return G, E

    elseif expr.head == :comprehension
        @assert length(expr.args) == 1
        @assert expr.args[1].head == :generator
        gen = expr.args[1]
        body = gen.args[1]
        loop = gen.args[2] # has to be static
        @assert loop.head == :(=)
        
        loop_var = loop.args[1]
        range = loop.args[2]
        G_range, range = transpile(t, phi, range)  
        @assert isempty(G_range.V)

        G = EmptyPGM()
        E = []
        for i in eval(range)
            e = substitute(loop_var, i, body)
            Gi, Ei = transpile(t, phi, e)
            G = graph_disjoint_union(G, Gi)
            push!(E, Ei)
        end
        return G, Expr(:vect, E...)

    elseif expr.head == :sample
        addr = eval(expr.args[1])
        dist = expr.args[2]
        G, E = transpile(t, phi, dist)

        # generate fresh variable
        v = gensym(:sample)
        push!(t.variables, v)
        push!(t.all_let_variables, v)
        t.variable_to_address[v] = addr

        # should only contain vars from sample or observe
        # all other vars should be substituted for their let value
        free_vars = get_free_variables(E) ∩ t.all_let_variables # remove primitives from Julia
        for z in free_vars
            @assert z in t.variables
            push!(G.A, z => v)
        end
        push!(G.V, v)
    
        G.P[v] = E # distribution not score
    
        return G, v

    elseif expr.head == :observe
        dist = expr.args[1]
        observation = expr.args[2]
        G1, E1 = transpile(t, phi, dist)
        G2, E2 = transpile(t, phi, observation)
        G = graph_disjoint_union(G1, G2)
    
        # generate fresh variable
        v = gensym(:observe)
        push!(t.variables, v)
        push!(t.all_let_variables, v)

        F = :($phi ? $E1 : true) # TODO: relpace with "flat" distribution
        G.P[v] = simplify_if(F)  # distribution not score
    
        free_vars = get_free_variables(F) ∩ t.all_let_variables # remove primitives from Julia
        for z in free_vars
            z == v && continue # necessary
            @assert z in t.variables
            push!(G.A, z => v)
        end
    
        push!(G.V, v)
    
        @assert length(get_free_variables(E2) ∩ t.all_let_variables) == 0
    
        G.Y[v] = E2
    
        return G, E2
    elseif expr.head == :$
        @assert length(expr.args) == 1
        v = eval(expr.args[1])
        if typeof(v) <: AbstractVector
            return EmptyPGM(), Expr(:vect, v...)
        else
            return EmptyPGM(), v # v has to be literal (bool, float, etc.)
        end
    else
        println(expr.args)
        error("Unsupported expression $(expr.head).")
    end
end

function logpdf(b::Bool, x::Float64)
    return 0.
end

function transpile_program(expr::Expr)
    t = PGMTranspiler()
    all_let_variables = get_let_variables(expr)
    # println("all_let_variables:", all_let_variables)
    if !isempty(all_let_variables)
        push!(t.all_let_variables, all_let_variables...)
    end

    main = expr.args[end]
    if length(expr.args) > 1
        for f in expr.args[1:end-1]
            @assert f.head == :function f.head
            f_def = f.args[1]
            f_name = f_def.args[1]
            t.procs[f_name] = f
        end
    end
    

    G, E = transpile(t, true, main)
    return G, E, t.variable_to_address
end

struct PGM
    name::Symbol
    n_variables::Int
    edges::Set{Pair{Int,Int}} # edges
    addresses::Vector{Any}
    distributions::Vector{Function} # distributions not pdfs
    observed_values::Vector{Union{Nothing,Function}} # observations
    return_expr::Function
    sample::Function
    logpdf::Function
    sym_to_ix::Dict{Symbol, Int}
    symbolic_pgm::SymbolicPGM
    symbolic_return_expr::Union{Expr, Symbol}
    topological_order::Vector{Int}
end

function Base.show(io::IO, pgm::PGM)
    spgm, E = to_human_readable(pgm.symbolic_pgm, pgm.symbolic_return_expr, pgm.sym_to_ix)
    println(io, pgm.name)
    println(io, spgm)
    println(io, "Return expression:")
    println(io, E)
    println(io, "Addresses:")
    for (i, addr) in enumerate(pgm.addresses)
        if !isnothing(addr)
            println(io, "x$i -> ", addr)
        end
    end
    println(io, "Topological Order:")
    println(io, pgm.topological_order)
end

function get_topolocial_order(n_variables::Int, edges::Set{Pair{Int,Int}})
    roots = [i for i in 1:n_variables if !any(i == y for (x,y) in edges)]
    ordered_nodes = Int[] # topological order
    nodes = roots
    while length(nodes) > 0
        node = popfirst!(nodes)
        children = [y for (x,y) in edges if x == node]
        push!(nodes, children...)
        if !(node in ordered_nodes)
            push!(ordered_nodes, node)
        end
    end
    return ordered_nodes
end

function human_readable_symbol(spgm, sym, j)
    return haskey(spgm.Y, sym) ? Symbol("y$j") : Symbol("x$j")
end

function to_human_readable(spgm::SymbolicPGM, E::Union{Expr, Symbol}, sym_to_ix)
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)

    new_spgm = EmptyPGM()
    for sym in spgm.V
        ix = sym_to_ix[sym]
        push!(new_spgm.V, human_readable_symbol(spgm, sym, ix))
    end
    for (x,y) in spgm.A
        i = sym_to_ix[x]
        j = sym_to_ix[y]
        push!(new_spgm.A, human_readable_symbol(spgm, x, i) => human_readable_symbol(spgm, y, j))
    end

    n_variables = length(spgm.V)
    for i in 1:n_variables
        sym = ix_to_sym[i]
        new_sym = human_readable_symbol(spgm, sym, i)
        
        d = spgm.P[sym]
        for j in 1:n_variables
            sub_sym = human_readable_symbol(spgm, ix_to_sym[j], j)
            d = substitute(ix_to_sym[j], sub_sym, d)
        end
        new_spgm.P[new_sym] = d

        if haskey(spgm.Y, sym)
            y = spgm.Y[sym]
            for j in 1:n_variables
                sub_sym = human_readable_symbol(spgm, ix_to_sym[j], j)
                y = substitute(ix_to_sym[j], sub_sym, y)
            end
            new_spgm.Y[new_sym] = y
        end
    end

    new_E = deepcopy(E)
    for j in 1:n_variables
        sub_sym = human_readable_symbol(spgm, ix_to_sym[j], j)
        new_E = substitute(ix_to_sym[j], sub_sym, new_E)
    end

    new_spgm, new_E
end

function compile_symbolic_pgm(name::Symbol, spgm::SymbolicPGM, E::Union{Expr, Symbol}, variable_to_address::Dict{Symbol, Any})
    n_variables = length(spgm.V)
    sym_to_ix = Dict(sym => ix for (ix, sym) in enumerate(
        sort(collect(spgm.V), lt=(x,y) -> haskey(spgm.Y,x) < haskey(spgm.Y,y)))  # observed nodes last
    )
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)
    edges = Set([sym_to_ix[x] => sym_to_ix[y] for (x, y) in spgm.A])
    
    lp = gensym(:lp)
    lp_block_args = []
    push!(lp_block_args, :($lp = 0.0))

    sample_block_args = []

    ordered = get_topolocial_order(n_variables, edges)
    @assert length(ordered) == n_variables

    X = gensym(:X)
    distributions = Vector{Function}(undef, n_variables)
    observed_values = Vector{Union{Nothing,Function}}(undef, n_variables)
    addresses = Vector{Any}(undef, n_variables)
    for i in ordered
        sym = ix_to_sym[i]
        d = spgm.P[sym]
        for j in 1:n_variables
            d = substitute(ix_to_sym[j], :($X[$j]), d)
        end
        f_name = Symbol("$(name)_dist_$i")
        f = rmlines(:(
            function $f_name($X::AbstractVector{Float64})
                $d
            end
        ))
        # display(f)
        distributions[i] = eval(f)

        d_sym = gensym("dist_$i")

        if haskey(spgm.Y, sym)
            y = spgm.Y[sym]
            # support dynamic obserations in general
            for j in 1:n_variables
                y = substitute(ix_to_sym[j], :($X[$j]), y)
            end
            f_name = Symbol("$(name)_obs_$i")
            f = rmlines(:(
                function $f_name($X::AbstractVector{Float64})
                    $y
                end
            ))
            # display(f)
            observed_values[i] = eval(f)
            addresses[i] = nothing
            push!(sample_block_args, :($X[$i] = $y))
        else
            observed_values[i] = nothing
            addresses[i] = variable_to_address[sym]
            push!(sample_block_args, :($d_sym = $d))
            push!(sample_block_args, :($X[$i] = rand($d)))
        end

        push!(lp_block_args, :($d_sym = $d))
        push!(lp_block_args, :($lp += logpdf($d_sym, $X[$i])))
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


    push!(lp_block_args, :($lp))

    f_name = Symbol("$(name)_logpdf")
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $(Expr(:block, lp_block_args...))
        end
    ))
    # display(f)
    logpdf = eval(f)

    f_name = Symbol("$(name)_return")
    new_E = deepcopy(E)
    for j in 1:n_variables
        new_E = substitute(ix_to_sym[j], :($X[$j]), new_E)
    end
    f = rmlines(:(
        function $f_name($X::AbstractVector{Float64})
            $new_E
        end
    ))
    # display(f)
    return_expr = eval(f)

    return PGM(
        name,
        n_variables, edges,
        addresses,
        distributions, observed_values,
        return_expr,
        sample,
        logpdf,
        sym_to_ix,
        spgm, E,
        ordered)
end


macro ppl(name, foppl)
    foppl = rmlines(foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, {symbol_} ~ dist_) ? Expr(:sample, symbol, dist) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, var_ ~ dist_) ? :($var = $(Expr(:sample, QuoteNode(var), dist))) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, dist_ ↦ s_) ? Expr(:observe, dist, s) : expr,  foppl);
    foppl = unwrap_let(foppl)
    
    G, E, variable_to_address = transpile_program(foppl);
    
    pgm = compile_symbolic_pgm(name, G, E, variable_to_address);

    return pgm
end

export @ppl