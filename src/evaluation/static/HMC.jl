import TinyPPL.Logjoint: hmc_logjoint

function hmc(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, eps::Float64)
    addresses_to_ix, logjoint, transform_to_constrained!, _ = make_unconstrained_logjoint(model, args, observations)
    K = length(addresses_to_ix)
    result = hmc_logjoint(logjoint, K, n_samples, L, eps)
    transform_to_constrained!(result)
    return Traces(addresses_to_ix, result)
end

export hmc