
import TinyPPL.Logjoint: hmc_logjoint
function hmc(pgm::PGM, n_samples::Int, L::Int, eps::Float64;
    ad_backend::Symbol=:tracker, x_initial::Union{Nothing,Vector{Float64}}=nothing, unconstrained::Bool=false)

    logjoint = unconstrained ? make_unconstrained_logjoint(pgm) : make_logjoint(pgm)

    # x_initial is constrained and is not mutated, default is a sample from the prior (which lies in the support)
    if isnothing(x_initial)
        x_initial = Vector{Float64}(undef, pgm.n_latents)
        pgm.sample!(x_initial)
    end
    if unconstrained
        x_initial = pgm.transform_to_unconstrained!(x_initial, similar(x_initial))
    else
        x_initial = copy(x_initial)
    end

    K = pgm.n_latents
    result = hmc_logjoint(logjoint, K, n_samples, L, eps, ad_backend=ad_backend, x_initial=x_initial)

    retvals = Vector{Any}(undef, n_samples)
    for i in 1:n_samples
        if unconstrained
            pgm.transform_to_constrained!(view(result, :,i), view(result,:,i))
        end
        retvals[i] = get_retval(pgm, result[:,i])
    end
    return GraphTraces(pgm, result, retvals)
end
export hmc
