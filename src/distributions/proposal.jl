"""
ProposalDistribution allows you to do conditional (e.g. RandomWalk) or
unconditional (e.g. StaticProposal) proposals with an unifed interface.
"""

abstract type ProposalDistribution end

function proposal_dist(dist::ProposalDistribution, x_current)
    error("Not implemented!")
end

function propose_and_logpdf(dist::ProposalDistribution, x_current)
    q = proposal_dist(dist, x_current)
    x_proposed = rand(q)
    return x_proposed, logpdf(q, x_proposed)
end

function proposal_logpdf(dist::ProposalDistribution, x_proposed, x_current)
    q = proposal_dist(dist, x_current)
    return logpdf(q, x_proposed)
end

export proposal_dist, propose_and_logpdf, proposal_logpdf

export ProposalDistribution

struct StaticProposal{D<:Distribution} <: ProposalDistribution
    base::D
end

function proposal_dist(dist::StaticProposal, x_current)
    return dist.base
end

export StaticProposal


# variance and bounds are settings of the inference algorithm (not differentiated), so Float64
struct ContinuousRandomWalkProposal <: ProposalDistribution
    variance::Float64
    lower::Float64
    upper::Float64
    function ContinuousRandomWalkProposal(variance::Real, lower::Real=-Inf, upper::Real=Inf)
        return new(variance, lower, upper)
    end
end

function proposal_dist(dist::ContinuousRandomWalkProposal, x_current)
    if dist.lower == -Inf && dist.upper == Inf
        proposer = Normal(x_current, sqrt(dist.variance))
    else
        proposer = ContinuousRWProposer(dist.lower, dist.upper, x_current, dist.variance)
    end
    return proposer
end

export ContinuousRandomWalkProposal

# upper = Inf for unbounded support
struct DiscreteRandomWalkProposal <: ProposalDistribution
    variance::Float64
    lower::Int
    upper::Float64
    function DiscreteRandomWalkProposal(variance::Real, lower::Real=0, upper::Real=Inf)
        return new(variance, lower, upper)
    end
end

function proposal_dist(dist::DiscreteRandomWalkProposal, x_current)
    if dist.upper == Inf
        proposer = DiscreteRWProposer(dist.lower, Inf, Int(x_current), dist.variance)
    else
        proposer = DiscreteRWProposer(dist.lower, Int(dist.upper), Int(x_current), dist.variance)
    end
    return proposer
end

export DiscreteRandomWalkProposal


# mapping addresses to proposal distributions
import TinyPPL: MostSpecificDict

const Addr2Proposal = MostSpecificDict{ProposalDistribution}

function Addr2Proposal(args...)
    return MostSpecificDict(Dict{Any, ProposalDistribution}(args...))
end

function Addr2Proposal(d::Dict{Any,T}) where T <: ProposalDistribution
    return MostSpecificDict{ProposalDistribution}(d)
end

export Addr2Proposal
