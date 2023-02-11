
const Proposal = MostSpecificDict{Distribution}

function Proposal(args...)
    return MostSpecificDict(Dict{Any, Distribution}(args...))
end

function Proposal(d::Dict{Any,T}) where T <: Distribution
    return MostSpecificDict{Distribution}(d)
end

const Addr2Var = MostSpecificDict{Float64}
function Addr2Var(args...)
    return MostSpecificDict(Dict{Any, Float64}(args...))
end

export Proposal, Addr2Var