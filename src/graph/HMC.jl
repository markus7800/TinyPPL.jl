
import TinyPPL.Logjoint: hmc_logjoint
function hmc(pgm::PGM, n_samples::Int, L::Int, eps::Float64)
    logjoint = make_logjoint(pgm)
    K = sum(isnothing.(pgm.observed_values))
    result = hmc_logjoint(logjoint, K, n_samples, L, eps)


    # TODO: enforce static observers, this is annoying
    Y = [pgm.observed_values[k](Vector{Float64}(undef, 0)) for k in (K+1):pgm.n_variables]

    for i in 1:n_samples
        Z = vcat(result[:,i], Y)
        result[:,i] = pgm.transform_to_constrained!(Z, Z)[1:K]
    end
    return result
end
export hmc
