
struct GraphVIResult <: VIResult
    Q::VariationalDistribution
    addresses_to_ix::Dict{Address,Int} # this do not have to be meaningful but often convienent
    pgm::PGM
    transform_to_constrained::Bool
end

function GraphVIResult(pgm::PGM, Q::VariationalDistribution, transform_to_constrained::Bool)
    addresses_to_ix = Dict{Any,Int}(pgm.addresses[i] => i for i in 1:pgm.n_latents)
    return GraphVIResult(Q, addresses_to_ix, pgm, transform_to_constrained)
end

Base.show(io::IO, viresult::GraphVIResult) = print(io, "GraphVIResult($(viresult.Q))")

function sample_posterior(res::GraphVIResult, n::Int)
    samples = rand(res.Q, n)
    @assert samples isa AbstractMatrix
    retvals = Vector{Any}(undef, n)
    @progress for i in axes(samples,2)
        X = view(samples,:,i)
        if res.transform_to_constrained
            res.pgm.transform_to_constrained!(X, X)
        end
        retvals[i] = get_retval(res.pgm, X)
    end
    return GraphTraces(res.addresses_to_ix, samples, retvals)
end
export sample_posterior


import TinyPPL.Logjoint: advi_meanfield_logjoint

function advi_meanfield(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint = make_unconstrained_logjoint(pgm)
    K = pgm.n_latents
    result = advi_meanfield_logjoint(logjoint, K, n_samples, L, learning_rate)
    return GraphVIResult(pgm, result, true)
end

import TinyPPL.Logjoint: advi_fullrank_logjoint

function advi_fullrank(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint = make_unconstrained_logjoint(pgm)
    K = pgm.n_latents
    result = advi_fullrank_logjoint(logjoint, K, n_samples, L, learning_rate)
    return GraphVIResult(pgm, result, true)
end

import TinyPPL.Logjoint: advi_logjoint
import TinyPPL.Distributions: ELBOEstimator, ReinforceELBO

function advi(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator; unconstrained::Bool=false)
    logjoint = unconstrained ? make_unconstrained_logjoint : make_logjoint(pgm)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    return GraphVIResult(pgm, result, unconstrained)
end

export advi_meanfield, advi_fullrank, advi


import TinyPPL.Distributions: MeanField, init_variational_distribution

# assumes static distributions types
function get_mixed_meanfield(pgm::PGM)::MeanField
    X = Vector{Float64}(undef, pgm.n_latents)
    pgm.sample!(X)
    K = pgm.n_latents
    dists = [init_variational_distribution(get_distribution(pgm, node, X)) for node in 1:K]
    return MeanField(dists)
end
export get_mixed_meanfield


function bbvi(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint = make_unconstrained_logjoint(pgm)
    q = get_mixed_meanfield(pgm)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, ReinforceELBO())
    return GraphVIResult(pgm, result, true)
end
export bbvi