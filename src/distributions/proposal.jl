"""
ProposalDistribution allows you to do conditional (e.g. RandomWalk) or
unconditional (e.g. StaticProposal) proposals with an unifed interface.
"""

abstract type ProposalDistribution end

function proposal_dist(dist::ProposalDistribution, x_current)
    error("Not implemented!")
end

function propose_and_logpdf(dist::ProposalDistribution, x_current)
    error("Not implemented!")
end

function proposal_logpdf(dist::ProposalDistribution, x_proposed, x_current)
    error("Not implemented!")
end

export proposal_dist, propose_and_logpdf, proposal_logpdf

export ProposalDistribution

struct StaticProposal <: ProposalDistribution
    base::Distribution
end

function proposal_dist(dist::StaticProposal, x_current)
    return dist.base
end

function propose_and_logpdf(dist::StaticProposal, x_current)
    x_proposed = rand(dist.base)
    return x_proposed, logpdf(dist.base, x_proposed)
end

function proposal_logpdf(dist::StaticProposal, x_proposed, x_current)
    return logpdf(dist.base, x_proposed)
end

export StaticProposal


struct ContinuousRandomWalkProposal <: ProposalDistribution
    variance::Real
    lower::Real
    upper::Real
    function ContinuousRandomWalkProposal(variance, lower=-Inf, upper=Inf)
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

function propose_and_logpdf(dist::ContinuousRandomWalkProposal, x_current)
    proposer = proposal_dist(dist, x_current)
    x_proposed = rand(proposer)
    return x_proposed, logpdf(proposer, x_proposed)
end

function proposal_logpdf(dist::ContinuousRandomWalkProposal, x_proposed, x_current)
    proposer = proposal_dist(dist, x_current)
    return logpdf(proposer, x_proposed)
end

export ContinuousRandomWalkProposal

struct DiscreteRandomWalkProposal <: ProposalDistribution
    variance::Real
    lower::Real
    upper::Real
    function DiscreteRandomWalkProposal(variance, lower=0, upper=Inf)
        return new(variance, lower, upper)
    end
end

function proposal_dist(dist::DiscreteRandomWalkProposal, x_current)
    if dist.upper == Inf
        proposer = DiscreteRWProposer(Int(dist.lower), Inf, Int(x_current), dist.variance)
    else
        proposer = DiscreteRWProposer(Int(dist.lower), Int(dist.upper), Int(x_current), dist.variance)
    end
    return proposer
end

function propose_and_logpdf(dist::DiscreteRandomWalkProposal, x_current)
    proposer = proposal_dist(dist, x_current)
    x_proposed = rand(proposer)
    return x_proposed, logpdf(proposer, x_proposed)
end

function proposal_logpdf(dist::DiscreteRandomWalkProposal, x_proposed, x_current)
    proposer = proposal_dist(dist, x_current)
    return logpdf(proposer, x_proposed)
end

export DiscreteRandomWalkProposal


# mapping addresses to proposal distributions

const Addr2Proposal = MostSpecificDict{ProposalDistribution}

function Addr2Proposal(args...)
    return MostSpecificDict(Dict{Any, ProposalDistribution}(args...))
end

function Addr2Proposal(d::Dict{Any,T}) where T <: ProposalDistribution
    return MostSpecificDict{ProposalDistribution}(d)
end

export Addr2Proposal
