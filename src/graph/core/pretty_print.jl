
function Base.show(io::IO, pgm::PGM)
    spgm, E = to_human_readable(pgm.symbolic_pgm, pgm.symbolic_return_expr, pgm.sym_to_ix)
    println(io, pgm.name)
    println(io, spgm)
    println(io, "Return expression:")
    println(io, E)
    println(io, "Addresses:")
    for (i, addr) in enumerate(pgm.addresses)
        if isobserved(pgm, i)
            println(io, "y$i -> ", addr)
        else
            println(io, "x$i -> ", addr)
        end
    end
    println(io, "Topological Order:")
    println(io, pgm.topological_order)
end


function human_readable_symbol(spgm, sym, j)
    n = ndigits(length(spgm.V))
    return haskey(spgm.Y, sym) ? Symbol("y"*lpad(j,n,"0")) : Symbol("x"*lpad(j,n,"0"))
end

function to_human_readable(spgm::SymbolicPGM, E::Union{Expr, Symbol, Real}, sym_to_ix)
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