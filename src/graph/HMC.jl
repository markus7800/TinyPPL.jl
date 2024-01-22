
import TinyPPL.Logjoint: hmc_logjoint
function hmc(pgm::PGM, n_samples::Int, L::Int, eps::Float64;
    ad_backend::Symbol=:tracker, x_initial::Union{Nothing,Vector{Float64}}=nothing, unconstrained::Bool=false)

    logjoint = unconstrained ? make_unconstrained_logjoint(pgm) : make_logjoint(pgm)

    if !isnothing(x_initial) && unconstrained
        pgm.transform_to_unconstrained!(x_initial)
    end


    K = pgm.n_latents
    result = hmc_logjoint(logjoint, K, n_samples, L, eps, ad_backend=ad_backend, x_initial=x_initial)
    # unconstrained logjoint automatically transforms to constrained

    retvals = Vector{Any}(undef, n_samples)
    for i in 1:n_samples
        retvals[i] = get_retval(pgm, result[:,i])
    end
    return GraphTraces(pgm, result, retvals)
end
export hmc
