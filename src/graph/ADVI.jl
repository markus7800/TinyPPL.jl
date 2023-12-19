import TinyPPL.Logjoint: advi_meanfield_logjoint

function get_number_of_latent_variables(pgm::PGM)
    return sum(isnothing.(pgm.observed_values))
end
export get_number_of_latent_variables


function advi_meanfield(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint = make_logjoint(pgm)
    K = get_number_of_latent_variables(pgm)
    result = advi_meanfield_logjoint(logjoint, K, n_samples, L, learning_rate)
    return result
end

import TinyPPL.Logjoint: advi_fullrank_logjoint

function advi_fullrank(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint = make_logjoint(pgm)
    K = get_number_of_latent_variables(pgm)
    result = advi_fullrank_logjoint(logjoint, K, n_samples, L, learning_rate)
    return result
end

import TinyPPL.Logjoint: advi_logjoint, ELBOEstimator, ReinforceELBO

function advi(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64, q::VariationalDistribution, estimator::ELBOEstimator)
    logjoint = make_logjoint(pgm)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, estimator)
    return result
end

export advi_meanfield, advi_fullrank, advi


import ..Distributions: MixedMeanField, init_variational_distribution

# assumes static distributions
function get_mixed_meanfield(pgm::PGM)::MixedMeanField
    X = Vector{Float64}(undef, pgm.n_variables)
    pgm.sample!(X)
    sample_mask = isnothing.(pgm.observed_values)
    K = sum(sample_mask)
    @assert all(sample_mask[1:K])

    dists = [init_variational_distribution(pgm.distributions[i](X)) for i in 1:K]
    return MixedMeanField(dists)
end
export get_mixed_meanfield


function bbvi(pgm::PGM, n_samples::Int, L::Int, learning_rate::Float64)
    logjoint = make_logjoint(pgm)
    q = get_mixed_meanfield(pgm)
    result = advi_logjoint(logjoint, n_samples, L, learning_rate, q, ReinforceELBO())
    return result
end
export bbvi