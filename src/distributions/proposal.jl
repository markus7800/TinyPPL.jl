
abstract type ProposalDistributions end

const Proposal = MostSpecificDict{ProposalDistributions}

function Proposal(args...)
    return MostSpecificDict(Dict{Any, ProposalDistributions}(args...))
end

function Proposal(d::Dict{Any,T}) where T <: ProposalDistributions
    return MostSpecificDict{ProposalDistributions}(d)
end

const Addr2Var = MostSpecificDict{Float64}
function Addr2Var(args...)
    return MostSpecificDict(Dict{Any, Float64}(args...))
end

export Proposal, Addr2Var


function propose_and_logpdf(dist::ProposalDistributions, x_current)
    error("Not implemented!")
end

function proposal_logpdf(dist::ProposalDistributions, x_proposed, x_current)
    error("Not implemented!")
end

export ProposalDistributions

struct StaticProposal <: ProposalDistributions
    base::Distribution
end

function propose_and_logpdf(dist::StaticProposal, x_current)
    x_proposed = rand(dist.base)
    return x_proposed, logpdf(dist.base, x_proposed)
end

function proposal_logpdf(dist::StaticProposal, x_proposed, x_current)
    return logpdf(dist.base, x_proposed)
end

export StaticProposal


struct ContinuousRandomWalkProposal <: ProposalDistributions
    variance::Real
    lower::Real
    upper::Real
    function ContinuousRandomWalkProposal(variance, lower=-Inf, upper=Inf)
        return new(variance, lower, upper)
    end
end

function propose_and_logpdf(dist::ContinuousRandomWalkProposal, x_current)
    if dist.lower == -Inf && dist.upper == Inf
        proposer =  Normal(x_current, sqrt(dist.variance))
    else
        proposer = ContinuousRWProposer(dist.lower, dist.upper, x_current, dist.variance)
    end
    x_proposed = rand(proposer)
    return x_proposed, logpdf(proposer, x_proposed)
end

function proposal_logpdf(dist::ContinuousRandomWalkProposal, x_proposed, x_current)
    if dist.lower == -Inf && dist.upper == Inf
        proposer =  Normal(x_current, sqrt(dist.variance))
    else
        proposer = ContinuousRWProposer(dist.lower, dist.upper, x_current, dist.variance)
    end
    return logpdf(proposer, x_proposed)
end

export ContinuousRandomWalkProposal

struct DiscreteRandomWalkProposal <: ProposalDistributions
    variance::Real
    lower::Real
    upper::Real
    function DiscreteRandomWalkProposal(variance, lower=0, upper=Inf)
        return new(variance, lower, upper)
    end
end

function propose_and_logpdf(dist::DiscreteRandomWalkProposal, x_current)
    if dist.upper == Inf
        proposer = DiscreteRWProposer(Int(dist.lower), Inf, Int(x_current), dist.variance)
    else
        proposer = DiscreteRWProposer(Int(dist.lower), Int(dist.upper), Int(x_current), dist.variance)
    end
    x_proposed = rand(proposer)
    return x_proposed, logpdf(proposer, x_proposed)
end

function proposal_logpdf(dist::DiscreteRandomWalkProposal, x_proposed, x_current)
    if dist.upper == Inf
        proposer = DiscreteRWProposer(Int(dist.lower), Inf, Int(x_current), dist.variance)
    else
        proposer = DiscreteRWProposer(Int(dist.lower), Int(dist.upper), Int(x_current), dist.variance)
    end
    return logpdf(proposer, x_proposed)
end

export DiscreteRandomWalkProposal
