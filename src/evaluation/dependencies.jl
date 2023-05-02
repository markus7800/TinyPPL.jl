


# struct Call{F,As<:Tuple}
#     func::F
#     args::As
# end

# mutable struct Tracked{T}
#     val::T
#     parent::Union{Nothing, Tracked}
#     function Tracked(val::T) where T
#         return new{T}(val, nothing)
#     end
# end

# function track(f::F, xs...; kw...) where F

# end


mutable struct DependencyAnalyzer <: Sampler
    dependencies::Dict
    function DependencyAnalyzer()
        return new(Dict())
    end
end

struct TrackedValue{T} <: Real where T <: Real
    val::T
    deps::Set
    function TrackedValue(val::T, deps=Set()) where T <: Real
        return new{T}(val, deps)
    end
end
export TrackedValue

Base.convert(::Type{TrackedValue{T}}, x::TrackedValue{S}) where {T <: Real, S <: Real} = TrackedValue(convert(T, x.val), x.deps)
Base.convert(::Type{TrackedValue{T}}, x::Real) where T <: Real = TrackedValue(convert(T, x))

Base.promote_rule(::Type{TrackedValue{T}}, ::Type{TrackedValue{S}}) where {T <: Real, S <: Real} = TrackedValue{promote_type(T,S)}
Base.promote_rule(::Type{TrackedValue{T}}, ::Type{S}) where {T <: Real, S <: Real} = TrackedValue{T}

Base.:<(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = x.val < y.val
Base.:(==)(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = x.val == y.val
# Base.hash(x::TrackedValue{T}) where T <: Real = hash(x.val)
# Base.hash(x::TrackedValue{T}, h::UInt) where T <: Real = hash(x.val, h)

Base.:+(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = TrackedValue(x.val + y.val, x.deps ∪ y.deps)
Base.:*(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = TrackedValue(x.val * y.val, x.deps ∪ y.deps)
Base.:-(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = TrackedValue(x.val - y.val, x.deps ∪ y.deps)
Base.:/(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = TrackedValue(x.val / y.val, x.deps ∪ y.deps)

Base.:-(x::TrackedValue{T}) where T <: Real = TrackedValue(-x.val, x.deps)

Base.isfinite(x::TrackedValue{T}) where T <: Real = isfinite(x.val)
Base.isapprox(x::TrackedValue{T}, y::TrackedValue{T}) where T <: Real = isapprox(x.val, y.val)

Base.Integer(x::TrackedValue{T}) where T <: Real = Integer(x.val)
Base.float(x::TrackedValue{T}) where T <: Real = float(x.val)
# Base.decompose(x::TrackedValue{T}) where T <: Real = Base.decompose(x.val)

Base.floor(x::TrackedValue{T}) where T <: Real = TrackedValue(floor(x.val), x.deps)
Base.ceil(x::TrackedValue{T}) where T <: Real = TrackedValue(ceil(x.val), x.deps)
Base.sqrt(x::TrackedValue{T}) where T <: Real = TrackedValue(sqrt(x.val), x.deps)

function Base.getindex(A::AbstractArray{T}, I::TrackedValue{Int64}) where T <: Real
    deps = I.deps
    for v in A
        if v isa TrackedValue
            deps = deps ∪ v.deps
        end
    end
    A[I.val] isa TrackedValue ? TrackedValue(A[I.val].val, deps) : TrackedValue(A[I.val], deps)
end

    # Base.setindex!(A::AbstractArray{T}, V::TrackedValue{S}, I::TrackedValue{Int64}) where {T <: Real,S <: Real} = A[I.val] = TrackedValue(V.val, I.deps ∪ V.deps)
# Base.setindex!(A::AbstractArray{T}, V, I::TrackedValue{Int64}) where T <: Real = A[I.val] = TrackedValue(V, I.deps)


import Distributions: params
function sample(sampler::DependencyAnalyzer, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    println(addr, ": ", dist)
    deps = Set()
    for p in params(dist)
        if p isa TrackedValue
            deps = deps ∪ p.deps
        elseif p isa AbstractArray{<:TrackedValue}
            for el in p
                deps = deps ∪ el.deps
            end
        end
    end
    println("\t", deps, " => ", addr)

    for parent in deps
        children = get!(sampler.dependencies, parent, Set())
        push!(children, addr)
    end

    if !isnothing(obs)
        return obs
    end

    return TrackedValue(rand(dist), Set(Any[addr]))
end

export DependencyAnalyzer