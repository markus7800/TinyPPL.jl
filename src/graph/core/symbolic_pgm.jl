
struct SymbolicPGM
    V::Set{Symbol} # vertices
    A::Set{Pair{Symbol,Symbol}} # edges
    P::Dict{Symbol, Any} # distributions not pdfs
    Y::Dict{Symbol, Float64} # observations
end

isobserved(spgm::SymbolicPGM, sym::Symbol) = haskey(spgm.Y, sym)

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
        Dict{Symbol, Float64}(),
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

#=
    Recursively transforms an expression with sample and observe statements
    to symbolic probabilistic graphical model (PGM) and an return expression.

    For each sample and observe expression a new node is generated and the
    corresponding distribution, observed value and dependencies are saved.

    For instance, for an if statement
    if condition
        holds
    else
        otherwise
    end
    we first transpile condition, holds and otherwise to a PGMs G1, G2, G3 and expressions
    E1, E2, E3, then join the (necessarily disjoint) PGMs G1 ∪ G2 ∪ G3 and build new return
    expression
    if E1
        E2
    else
        E3
    end

    We also update `phi`, which tracks the control flow, such that observe statement
    in unexecuted branches do not contribute to the log-probability.

    Allowed expressions:
    Blocks:
        begin
            statement_1
            statement_2
            ...
        end
    Single assignment lets:
        let x = 1
            body
        end
    If expressions:
        if condition
            holds
        else
            otherwise
        end
        condition ? holds : otherwise
    Function calls:
        func(arg_1, arg_2, ..., arg_n)
    Vector literals:
        [e_1, e_2, ..., e_n]
    Tuple literals:
        (e_1, e_2, ..., e_n)
    Vector indexing (but not setting):
        V[i]
    Loop comprehension with static range:
        [e(i) for i in range]
    Quoted symbols for inlining "out-of-block" data
        $(Main.X)
        where X is data in the main environment (bool, integer, float, vector, ...)
    Sample and observe expressions:
        Expr(:sample, address, distribution)
        Expr(:observe, address, distribution, value)
        where address is a static address.
=#
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

    elseif expr.head == :if ||  expr.head == :elseif
        @assert 2 ≤ length(expr.args) ≤ 3
        condition = expr.args[1]
        holds = expr.args[2]
        G1, E1 = transpile(t, phi, condition)
        G2, E2 = transpile(t, :($phi && $E1), holds)
        G = graph_disjoint_union(G1, G2)
        if length(expr.args) == 3
            otherwise = expr.args[3]
            G3, E3 = transpile(t, :($phi && !($E1)), otherwise)
            G = graph_disjoint_union(G, G3)
            E = Expr(expr.head, E1, E2, E3)
        else
            E = Expr(expr.head, E1, E2)
        end
        return G, E

    elseif expr.head == :call
        name = expr.args[1]
        @assert name isa Symbol || (name isa Expr && name.head == :.) name
        arguments = expr.args[2:end]
        res = [transpile(t, phi, arg) for arg in arguments]
        Gs = [r[1] for r in res]
        Es = [r[2] for r in res]

        G = EmptyPGM()
        for Gi in Gs
            G = graph_disjoint_union(G, Gi)
        end

        Eh = name

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

    elseif expr.head in [:vect, :tuple, :&&, :||]
        G = EmptyPGM()
        E = []
        for v in expr.args
            Gv, Ev = transpile(t, phi, v)
            G = graph_disjoint_union(G, Gv)
            push!(E, Ev)
        end
        return G, Expr(expr.head, E...)

    elseif expr.head == :ref
        @assert length(expr.args) == 2

        G_arr, arr = transpile(t, phi, expr.args[1])
        if expr.args[2] isa Int && (arr.head == :vect || arr.head == :tuple)
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
        addr = eval(expr.args[1])
        dist = expr.args[2]
        observation = expr.args[3]

        G1, E1 = transpile(t, phi, dist)
        # G2, E2 = transpile(t, phi, observation)
        # G = graph_disjoint_union(G1, G2)
        G = G1
        @assert isempty(get_free_variables(observation) ∩ t.all_let_variables) "$observation is not static."
        E2 = eval(observation) # static observations
    
        # generate fresh variable
        v = gensym(:observe)
        push!(t.variables, v)
        push!(t.all_let_variables, v)
        t.variable_to_address[v] = addr

        F = :($phi ? $E1 : Flat())
        G.P[v] = simplify_if(F)  # distribution not score
    
        free_vars = get_free_variables(F) ∩ t.all_let_variables # remove primitives from Julia
        for z in free_vars
            z == v && continue # necessary
            @assert z in t.variables
            push!(G.A, z => v)
        end
    
        push!(G.V, v)
    
        # @assert length(get_free_variables(E2) ∩ t.all_let_variables) == 0
    
        G.Y[v] = E2
        
        # we return value not symbol, values are injected into source
        # typicall these values are not used anywhere
        # it is best practice not to use them (not bind an observe statement to a variable in let)
        # alternative implemenation: make observations also symbolic:
        # adv: oberved values are not compiled into distributions, we can use a different set of observed values after compilation
        # disadv: we have to index into observations in distributions
        return G, E2

    elseif expr.head == :$
        @assert length(expr.args) == 1
        v = eval(expr.args[1])
        if typeof(v) <: AbstractVector
            return EmptyPGM(), Expr(:vect, v...)
        else
            return EmptyPGM(), v # v has to be literal (bool, float, etc.)
        end
    elseif expr.head == :macrocall
        # expr.args[2] === nothing
        if expr.args[1] == Symbol("@iterate")
            # println("@iterate ", expr.args)
            # @iterate(count, f, init)
            count = expr.args[3]
            func = expr.args[4]
            init = expr.args[5]
            new_expr = repeatf_symbolic(count, func, init)
            return transpile(t, phi, new_expr)
        elseif expr.args[1] == Symbol("@loop")
            # println("@loop ", expr.args)
            # @loop(count, f, init, args...)
            count = expr.args[3]
            func = expr.args[4]
            init = expr.args[5]
            args = expr.args[6:end]
            
            new_expr = loopf_symbolic(count, func, init, args)
            return transpile(t, phi, new_expr)
        else
            error("Unsupported macro $(expr.args[1])")
        end
    else
        println(expr.args)
        error("Unsupported expression $(expr.head).")
    end
end

struct Flat end
function logpdf(::Flat, x::Real)
    return 0.
end

function transpile_program(expr::Expr)
    t = PGMTranspiler()
    all_let_variables = get_let_variables(expr)
    
    if !isempty(all_let_variables)
        for var in all_let_variables
            push!(t.all_let_variables, var)
        end
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