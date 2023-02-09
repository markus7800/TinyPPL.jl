
import ..TinyPPL.Distributions: Distribution

function reverse_pair(pair::Pair)
    new_pair = pair[1]
    while pair[2] isa Pair
        pair = pair[2]
        new_pair = new_pair => pair[1]
    end
    new_pair => pair[2]
end

struct Proposal
    Q::Dict{Any, Distribution}
    function Proposal(Q::Dict{Any, T}) where T <: Distribution
        Q_new = Dict{Any, Distribution}()
        for (k, d) in Q
            if k isa Pair
                # reverse order, we want to be able to go from most specific to most general
                # z => i => j, try z=>i=>j, then z=>i, then z
                Q_new[reverse_pair(k)] = d
            else
                Q_new[k] = d
            end
        end 
        return new(Q_new)
    end
end

function Proposal(args...)
    return Proposal(Dict{Any, Distribution}(args...))
end

function Base.get(p::Proposal, k::Any, default::Distribution)
    return get(p.Q, k, default)
end

function Base.get(p::Proposal, k::Pair, default::Distribution)
    if haskey(p.Q, k)
        return p.Q[k]
    end
    # go from most specific to most general
    # z => i => j, try z=>i=>j, then z=>i, then z
    reversed_pair = reverse_pair(k)
    while reversed_pair isa Pair
        if haskey(p.Q, reversed_pair[1])
            return p.Q[reversed_pair[1]]
        else
            reversed_pair = reversed_pair[1]
        end
    end
    return get(p.Q, reversed_pair, default)
end

export Proposal