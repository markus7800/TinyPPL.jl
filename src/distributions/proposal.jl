
const Proposal = MostSpecificDict{Distribution}

function Proposal(args...)
    return MostSpecificDict(Dict{Any, Distribution}(args...))
end

function Proposal(d::Dict{Any,T}) where T <: Distribution
    return MostSpecificDict{Distribution}(d)
end

export Proposal