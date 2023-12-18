
import TinyPPL.Logjoint: hmc_logjoint
function hmc(pgm::PGM, n_samples::Int, L::Int, eps::Float64)
    logjoint = make_logjoint(pgm)
    K = sum(isnothing.(pgm.observed_values))
    return hmc_logjoint(logjoint, K, n_samples, L, eps)
end
export hmc
