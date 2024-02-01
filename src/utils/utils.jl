import MacroTools

rmlines(expr) = MacroTools.postwalk(sub_ex -> MacroTools.rmlines(sub_ex), expr)

isbraced(expr) = expr isa Expr && expr.head == :braces

function normalise(logprobs::Vector{Float64})
    m = maximum(logprobs)
    l = m + log(sum(exp, logprobs .- m))
    return logprobs .- l
end

function logsumexp(x)
    m = maximum(x)
    if m == -Inf
        return -Inf
    end
    return log(sum(exp, x .- m)) + m
end

sigmoid(x) = 1 / (1 + exp(-x))
function ∇sigmoid(x)
    ex = exp(x)
    dn = (ex + 1)^2
    return ex/dn
end
invsigmoid(x) = log(x / (1-x))

no_grad(x::Float64) = x
no_grad(x::Vector{Float64}) = x
no_grad(x::Dict{<:Any,Vector{Float64}}) = x

import Tracker
no_grad(x::Tracker.TrackedReal{Float64}) = Tracker.data(x)
no_grad(x::Tracker.TrackedVector{Float64, Vector{Float64}}) = Tracker.data(x)
no_grad(x::Tracker.TrackedMatrix{Float64, Matrix{Float64}}) = Tracker.data(x)
no_grad(x::Vector{Tracker.TrackedReal{Float64}}) = Tracker.data.(x)
no_grad(x::Matrix{Tracker.TrackedReal{Float64}}) = Tracker.data.(x)
no_grad(x::Dict{K,Tracker.TrackedVector{Float64, Vector{Float64}}}) where {K <: Any} = Dict{K,Vector{Float64}}(addr => no_grad(v) for (addr, v) in x)

import ForwardDiff
no_grad(x::ForwardDiff.Dual) = ForwardDiff.value(x)
no_grad(x::Vector{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)

import ReverseDiff
no_grad(x::ReverseDiff.TrackedReal) = ReverseDiff.value(x)
no_grad(x::ReverseDiff.TrackedArray) = ReverseDiff.value(x)

import Base, Random
Base.randn(::Type{ReverseDiff.TrackedReal{V,D,O}}) where {V,D,O} = ReverseDiff.TrackedReal{V,D,O}(randn(V))
Base.randn(rng::Random.AbstractRNG, ::Type{ReverseDiff.TrackedReal{V,D,O}}) where {V,D,O} = ReverseDiff.TrackedReal{V,D,O}(randn(rng, V))

# maps :x => :y => :z = :x => (:y => :z) to (:x => :y) => :z
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
        if haskey(d, reversed_pair)
            return d[reversed_pair]
        else
            reversed_pair = reversed_pair[1]
        end
    end
    return get(d, reversed_pair, default)
end

function has_match(d::Dict{Any, T}, k::Pair) where T
    reversed_pair = reverse_pair(k)
    while reversed_pair isa Pair
        if haskey(d, reversed_pair)
            return true
        else
            reversed_pair = reversed_pair[1]
        end
    end
    return haskey(d, reversed_pair)
end

function get_most_specific_match(d::Dict{Any, T}, k::Pair) where T
    reversed_pair = reverse_pair(k)
    while reversed_pair isa Pair
        if haskey(d, reversed_pair)
            return d[reversed_pair]
        else
            reversed_pair = reversed_pair[1]
        end
    end
    return d[reversed_pair]
end

"""
Dictionairy d for keys of form
    key ::= const
    key ::= key => const
if key => const is not in dict, it will try to lookup key.
E.g. if you have addresses of form :x => i, i = 1..N
Then, you can map the same object to every :x => i, by inserting d[:x] = object
Since :x => :y => :z = :x => (:y => :z) we reverse the pair to (:x => :y) => :z,
before inserting in the underlying dict Q and before looking up the key.
So the lookup strategy works with multi-level addresses.
"""
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

function Base.setindex!(d::MostSpecificDict{T}, v::T, k::Pair) where T
    d.Q[reverse_pair(k)] = v
end
function Base.setindex!(d::MostSpecificDict{T}, v::T, k) where T
    d.Q[k] = v
end
function Base.get(d::MostSpecificDict{T}, k::Any, default::T) where T
    return get(d.Q, k, default)
end

function Base.getindex(d::MostSpecificDict{T}, k::Any) where T
    return d.Q[k]
end

function Base.get(d::MostSpecificDict{T}, k::Pair, default::T) where T
    if haskey(d.Q, k)
        return d.Q[k]
    end
    return get_most_specific_match(d.Q, k, default)
end

function Base.getindex(d::MostSpecificDict{T}, k::Pair) where T
    if haskey(d.Q, k)
        return d.Q[k]
    end
    return get_most_specific_match(d.Q, k)
end

function Base.haskey(d::MostSpecificDict{T}, k::Any) where T
    return haskey(d.Q, k)
end

function Base.haskey(d::MostSpecificDict{T}, k::Pair) where T
    if haskey(d.Q, k)
        return true
    end
    return has_match(d.Q, k)
end
