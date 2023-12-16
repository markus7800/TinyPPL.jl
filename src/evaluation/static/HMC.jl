import TinyPPL.Logjoint: hmc_logjoint

function hmc(model::StaticModel, args::Tuple, observations::Dict, n_samples::Int, L::Int, eps::Float64)
    ulj = make_unconstrained_logjoint(model, args, observations)
    K = length(ulj.addresses_to_ix)
    result = hmc_logjoint(ulj.logjoint, K, n_samples, L, eps)
    ulj.transform_to_constrained!(result)
    return Traces(ulj.addresses_to_ix, result)
end

export hmc