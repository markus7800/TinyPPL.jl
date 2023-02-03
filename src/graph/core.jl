import MacroTools
import Distributions

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
                @assert length(bindings.args) > 1

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


# Computes the Probabilistic Graphical Model of a FOPPL program.

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
    proc_names::Set{Symbol}
    all_let_variables::Set{Symbol}
    variables::Set{Symbol}
    function PGMTranspiler()
        return new(Dict{Symbol, Expr}(), Set{Symbol}(), Set{Symbol}(), Set{Symbol}())
    end
end

function transpile(t::PGMTranspiler, phi, expr::Symbol)
    return EmptyPGM(), expr
end

function transpile(t::PGMTranspiler, phi, expr::Real)
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

        if Eh in t.proc_names
            # TODO: procedures
            # @assert Eh in t.proc_names Eh
            # proc = t.procs[Eh]
            # e_proc = proc.body
            # @assert length(proc.args) == length(Es)
            # for (arg, Ei) in zip(proc.args, Es)
            #     e_proc = substitute(arg, Ei, e_proc)
            # end
            # G_proc, E = transpile(t, phi, e_proc)
            # G = graph_disjoint_union(G, G_proc)
            # return G_proc, E
        end

        E = Expr(:call, Eh, Es...)
        return G, E

    elseif expr.head == :sample
        dist = expr.args[1]
        G, E = transpile(t, phi, dist)

        # generate fresh variable
        v = gensym(:sample)
        push!(t.variables, v)
        push!(t.all_let_variables, v)

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
        G.P[v] = F  # distribution not score
    
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
    else
        println(expr.args)
        error("Unsupported expression $(expr.head).")
    end
end

function Distributions.logpdf(b::Bool, x::Float64)
    return 0.
end

function transpile_program(expr::Expr)
    t = PGMTranspiler()
    all_let_variables = get_let_variables(expr)
    # println("all_let_variables:", all_let_variables)
    if !isempty(all_let_variables)
        push!(t.all_let_variables, all_let_variables...)
    end
    return transpile(t, true, expr)
end

struct PGM
    n_variables::Int
    edges::Set{Pair{Int,Int}} # edges
    distributions::Vector{Function} # distributions not pdfs
    observed_values::Vector{Union{Nothing,Function}} # observations
    return_expr::Function
    symbolic_pgm::SymbolicPGM
    symbolic_return_expr::Union{Expr, Symbol}
    topological_order::Vector{Int}
end

function Base.show(io::IO, pgm::PGM)
    println(io, pgm.symbolic_pgm)
    println(io, "Return expression")
    println(pgm.symbolic_return_expr)
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

function to_human_readable(spgm::SymbolicPGM, E::Union{Expr, Symbol}, ix_to_sym, sym_to_ix)
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

    new_E = copy(E)
    for j in 1:n_variables
        sub_sym = human_readable_symbol(spgm, ix_to_sym[j], j)
        new_E = substitute(ix_to_sym[j], sub_sym, new_E)
    end

    new_spgm, new_E
end

function compile_symbolic_pgm(spgm::SymbolicPGM, E::Union{Expr, Symbol})
    n_variables = length(spgm.V)
    sym_to_ix = Dict(sym => ix for (ix, sym) in enumerate(spgm.V))
    ix_to_sym = Dict(ix => sym for (sym, ix) in sym_to_ix)
    edges = Set([sym_to_ix[x] => sym_to_ix[y] for (x, y) in spgm.A])
    
    X = gensym(:X)
    distributions = []
    observed_values = []
    for i in 1:n_variables
        sym = ix_to_sym[i]
        d = spgm.P[sym]
        for j in 1:n_variables
            d = substitute(ix_to_sym[j], :($X[$j]), d)
        end
        f_name = Symbol("dist_$i")
        f = rmlines(:(
            function $f_name($X)
                $d
            end
        ))
        # display(f)
        push!(distributions, Main.eval(f))

        if haskey(spgm.Y, sym)
            y = spgm.Y[sym]
            for j in 1:n_variables
                y = substitute(ix_to_sym[j], :($X[$j]), y)
            end
            f_name = Symbol("obs_$i")
            f = rmlines(:(
                function $f_name($X)
                    $y
                end
            ))
            # display(f)
            push!(observed_values, Main.eval(f))
        else
            push!(observed_values, nothing)
        end
    end

    f_name = Symbol("return")
    symbolic_E = copy(E)
    for j in 1:n_variables
        E = substitute(ix_to_sym[j], :($X[$j]), E)
    end
    f = rmlines(:(
        function $f_name($X)
            $E
        end
    ))
    # display(f)
    return_expr = Main.eval(f)

    order = get_topolocial_order(n_variables, edges)
    spgm, E = to_human_readable(spgm, symbolic_E, ix_to_sym, sym_to_ix)
    return PGM(
        n_variables, edges,
        distributions, observed_values,
        return_expr,
        spgm, E,
        order)
end


macro pgm(foppl)
    foppl = rmlines(foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, s_ ~ dist_) ? :($s = $(Expr(:sample, dist))) : expr,  foppl);
    foppl = MacroTools.postwalk(expr -> MacroTools.@capture(expr, dist_ ↦ s_) ? Expr(:observe, dist, s) : expr,  foppl);
    foppl = unwrap_let(foppl)
    
    G, E = transpile_program(foppl);
    
    pgm = compile_symbolic_pgm(G, E);

    return pgm
end

export @pgm