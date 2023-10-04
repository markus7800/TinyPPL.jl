import ..Distributions: Transform, transform_to, to_unconstrained, support

# struct TransformCollector <: StaticSampler
#     addresses_to_transforms::Dict{Any,Transform}
#     function TransformCollector()
#         return new(Dict())
#     end
# end

# function sample(sampler::TransformCollector, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
#     if !isnothing(obs)
#         return obs
#     end
#     value = rand(dist)
#     sampler.addresses_to_transforms[addr] = transform_to(support(dist))
#     return value
# end

# function get_addresses_to_transforms(model::StaticModel, args::Tuple, observations::Dict)::Dict{Any,Transform}
#     sampler = TransformCollector()
#     model(args, sampler, observations)
#     return sampler.addresses_to_transforms
# end

# struct StackedTransforms
#     transforms::Vector{Transform}
#     addresses_to_ix::Addr2Ix
# end
# function StackedTransforms(addresses_to_ix::Addr2Ix, addresses_to_transform::Dict{Any,Transform})
#     transforms = Vector{Transform}(undef, length(addresses_to_transform))
#     for (addr, ix) in addresses_to_ix
#         transforms[ix] = addresses_to_transform[addr]
#     end
#     return StackedTransforms(transforms, addresses_to_ix)
# end
# function (st::StackedTransforms)(x::Vector{Float64})::Vector{Float64}
#     return Float64[st.transforms[i](x[i]) for i in eachindex(x)]
# end
# function Base.inv(st::StackedTransforms)::StackedTransforms
#     return StackedTransforms(inv.(st.transforms), st.addresses_to_ix)
# end
# function Base.getindex(st::StackedTransforms, addr::Any)::Transform
#     return st.transforms[st.addresses_to_ix[addr]]
# end
# function (st::StackedTransforms)(addr::Any, x::Float64)::Float64
#     return st[addr](x)
# end

mutable struct ConstraintTransformer{T} <: StaticSampler
    addresses_to_ix::Addr2Ix
    X::T
    Y::T
    to::Symbol
    function ConstraintTransformer(addresses_to_ix::Addr2Ix, X::T, Y::T; to::Symbol) where T <: AbstractVector{Float64}
        @assert to in (:constrained, :unconstrained)
        return new{T}(addresses_to_ix, X, Y, to)
    end
end 

function sample(sampler::ConstraintTransformer, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end
    transformed_dist = to_unconstrained(dist)
    i = sampler.addresses_to_ix[addr]
    if sampler.to == :unconstrained
        constrained_value = sampler.X[i]
        unconstrained_value = transformed_dist.T(constrained_value)
        sampler.Y[i] = unconstrained_value
    else # samper.to == :constrained
        unconstrained_value = sampler.X[i]
        constrained_value = transformed_dist.T_inv(unconstrained_value)
        sampler.Y[i] = constrained_value
    end
    return constrained_value
end

mutable struct UnconstrainedLogJoint{T,V} <: StaticSampler
    W::T
    addresses_to_ix::Addr2Ix
    X::V
    function UnconstrainedLogJoint(addresses_to_ix::Addr2Ix, X::V) where {T <: Real, V <: AbstractVector{T}}
        return new{eltype(V),V}(0., addresses_to_ix, X)
    end
end
function sample(sampler::UnconstrainedLogJoint, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        sampler.W += logpdf(dist, obs)
        return obs
    end
    unconstrained_value = sampler.X[sampler.addresses_to_ix[addr]]
    transformed_dist = to_unconstrained(dist)
    sampler.W += logpdf(transformed_dist, unconstrained_value)
    constrained_value = transformed_dist.T_inv(unconstrained_value)
    return constrained_value
end

function make_unconstrained_logjoint(model::StaticModel, args::Tuple, observations::Dict)
    addresses = get_addresses(model, args, observations)
    addresses_to_ix = get_address_to_ix(addresses)

    function transform_to_constrained!(X::AbstractArray{Float64})
        if ndims(X) == 2
            for i in axes(X,2)
                X_i = view(X, :, i)
                sampler = ConstraintTransformer(addresses_to_ix, X_i, X_i, to=:constrained)
                model(args, sampler, observations)
            end
        else
            sampler = ConstraintTransformer(addresses_to_ix, X, X, to=:constrained)
            model(args, sampler, observations)
        end
        return X
    end

    function transform_to_unconstrained!(X::AbstractArray{Float64})
        if ndims(X) == 2
            for i in axes(X,2)
                X_i = view(X, :, i)
                sampler = ConstraintTransformer(addresses_to_ix,  X_i, X_i, to=:unconstrained)
                model(args, sampler, observations)
            end
        else
            sampler = ConstraintTransformer(addresses_to_ix, X, X, to=:unconstrained)
            model(args, sampler, observations)
        end
        return X
    end

    function logjoint(X::AbstractVector{<:Real})
        sampler = UnconstrainedLogJoint(addresses_to_ix, X)
        model(args, sampler, observations)
        return sampler.W
    end

    return addresses_to_ix, logjoint, transform_to_constrained!, transform_to_unconstrained!
end