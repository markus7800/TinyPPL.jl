import Tracker

function make_logjoint(pgm::PGM)
    sample_mask = isnothing.(pgm.observed_values)
    n_sample = sum(sample_mask)
    @assert all(sample_mask[1:n_sample])
    @assert !any(sample_mask[n_sample+1:end])

    Z = Vector{Float64}(undef, pgm.n_variables)
    sample_mask = trues(pgm.n_variables)
    for node in pgm.topological_order
        d = pgm.distributions[node](Z)
        if !isnothing(pgm.observed_values[node])
            Z[node] = pgm.observed_values[node](Z) # observed values cannot depend on sampled values
        else
            Z[node] = rand(d) # sample from prior
        end
    end
    Y = Z[n_sample+1:end]

    function logjoint(X::AbstractVector{Float64})
        Z = vcat(X,Y)
        return pgm.logpdf(Z)
    end
    return logjoint
end

import TinyPPL.Logjoint: hmc_logjoint
function hmc(pgm::PGM, n_samples::Int, L::Int, eps::Float64)
    logjoint = make_logjoint(pgm)
    K = sum(isnothing.(pgm.observed_values))
    return hmc_logjoint(logjoint, K, n_samples, L, eps)
end
export hmc

# function hmc(pgm::PGM, n_samples::Int, ϵ::Float64, L::Int)
    
#     Z = Vector{Float64}(undef, pgm.n_variables)
#     sample_mask = trues(pgm.n_variables)
#     for node in pgm.topological_order
#         d = pgm.distributions[node](Z)
#         if !isnothing(pgm.observed_values[node])
#             Z[node] = pgm.observed_values[node](Z) # observed values cannot depend on sampled values
#             sample_mask[node] = false # X[.!sample_mask] is constant
#         else
#             Z[node] = rand(d) # sample from prior
#         end
#     end

#     function ∇U(X::Vector{Float64})::Vector{Float64} # input only sample sites
#         Z[sample_mask] = X
#         _Z = Tracker.param(Z) # we also get gradients at observed sites, but they are not used
#         lp = -pgm.logpdf(_Z) # we need to input sampled and observed sites to logpdf
#         Tracker.back!(lp)
#         return Tracker.grad(_Z)[sample_mask]
#     end

#     n_sample_sites = sum(sample_mask)
#     retvals = Vector{Any}(undef, n_samples)
#     logprobs = Vector{Float64}(undef, n_samples)
#     trace = Array{Float64,2}(undef, n_sample_sites, n_samples)
    
#     n_accepted = 0
#     X_current = copy(Z[sample_mask]) # only select sample sites
#     U_current = -pgm.logpdf(Z)
#     retval_current = pgm.return_expr(Z)
#     @progress for i in 1:n_samples
#         R = randn(n_sample_sites)
#         K_current = sum(R.^2) / 2
#         X_proposed, R_proposed = leapfog(∇U, R, copy(X_current), ϵ, L)

#         # X2, R2 = leapfog(∇U, R_proposed, copy(X_proposed), ϵ, L)
#         # X_current ≈ X2, R ≈ R2

#         Z[sample_mask] = X_proposed
#         U_proposed = -pgm.logpdf(Z)
#         K_proposed = sum(R_proposed.^2) / 2

#         if log(rand()) < U_current - U_proposed + K_current - K_proposed
#             retval_current = pgm.return_expr(Z)
#             U_current = U_proposed
#             X_current = X_proposed
#             n_accepted += 1
#         end

#         retvals[i] = retval_current
#         logprobs[i] = -U_current
#         trace[:,i] = X_current
#     end

#     @info "HMC" n_accepted/n_samples

#     return trace, retvals, logprobs
# end

# export hmc

# function leapfog(∇U::Function, R::Vector{Float64}, X::Vector{Float64}, ϵ::Float64, L::Int)::Tuple{Vector{Float64},Vector{Float64}}

#     R = R - ϵ/2 * ∇U(X) # half step

#     for _ in 1:(L-1)
#         X = X + ϵ * R # full step
#         R = R - ϵ * ∇U(X) # full step
#     end
#     X = X + ϵ * R # full step
    
#     R = R - ϵ/2 * ∇U(X) # half step

#     return X, -R
# end