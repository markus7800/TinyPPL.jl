import MacroTools

rmlines(expr) = MacroTools.postwalk(sub_ex -> MacroTools.rmlines(sub_ex), expr)

isbraced(expr) = expr isa Expr && expr.head == :braces

function normalise(logprobs::Vector{Float64})
    m = maximum(logprobs)
    l = m + log(sum(exp, logprobs .- m))
    return logprobs .- l
end


function reverse_pair(pair::Pair)
    new_pair = pair[1]
    while pair[2] isa Pair
        pair = pair[2]
        new_pair = new_pair => pair[1]
    end
    new_pair => pair[2]
end

function get_most_specific_match(d::Dict{Any, T}, k::Pair, default::T) where T
    # go from most specific to most general
    # z => i => j, try z=>i=>j, then z=>i, then z
    reversed_pair = reverse_pair(k)
    while reversed_pair isa Pair
        if haskey(d, reversed_pair[1])
            return d[reversed_pair[1]]
        else
            reversed_pair = reversed_pair[1]
        end
    end
    return get(d, reversed_pair, default)
end


struct MostSpecificDict{T}
    Q::Dict{Any, T}
    function MostSpecificDict(Q::Dict{Any, T}) where T
        Q_new = Dict{Any, T}()
        for (k, d) in Q
            if k isa Pair
                # reverse order, we want to be able to go from most specific to most general
                # z => i => j, try z=>i=>j, then z=>i, then z
                Q_new[reverse_pair(k)] = d
            else
                Q_new[k] = d
            end
        end 
        return new{T}(Q_new)
    end
end

function Base.get(d::MostSpecificDict{T}, k::Any, default::T) where T
    return get(d.Q, k, default)
end

function Base.get(d::MostSpecificDict{T}, k::Pair, default::T) where T
    if haskey(d.Q, k)
        return d.Q[k]
    end
    return get_most_specific_match(d.Q, k, default)
end