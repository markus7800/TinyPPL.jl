
using TinyPPL.Distributions
using TinyPPL.Graph
import Random

function get_meanfield_parameters(vi_result)
    mu = [d.base.μ for d in vi_result.Q.dists]
    sigma = [d.base.σ for d in vi_result.Q.dists]
    return mu, sigma    
end

begin
    # xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
    xs = [-1., -0.5, 0.0, 0.5, 1.0];
    ys = [-3.2, -1.8, -0.5, -0.2, 1.5];
    N = length(xs)


    slope_prior_mean = 0
    slope_prior_sigma = 3
    intercept_prior_mean = 0
    intercept_prior_sigma = 3

    σ = 2.0
    m0 = [0., 0.]
    S0 = [intercept_prior_sigma^2 0.; 0. slope_prior_sigma^2]
    Phi = hcat(fill(1., length(xs)), xs)
    map_Σ = inv(inv(S0) + Phi'Phi / σ^2) 
    map_mu = map_Σ*(inv(S0) * m0 + Phi'ys / σ^2)
    map_sigma = [sqrt(map_Σ[1,1]), sqrt(map_Σ[2,2])]
    println("map_mu")
    display(map_mu)
    println("map_Σ")
    display(map_Σ)
end


model = @pgm LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = $(Main.xs),
        ys = $(Main.ys),
        slope ~ Normal($(Main.slope_prior_mean), $(Main.slope_prior_sigma)),
        intercept ~ Normal($(Main.intercept_prior_mean), $(Main.intercept_prior_sigma))

        [({:y => i} ~ Normal(f(slope, intercept, xs[i]), $(Main.σ)) ↦ ys[i]) for i in 1:$(Main.N)]
        
        (slope, intercept)
    end
end

# try all inference algorithms

Random.seed!(0)
@time traces, lps = likelihood_weighting(model, 10^6)
W = exp.(lps);

W'traces[:intercept]
W'traces[:slope]


Random.seed!(0)
traces = lmh(model, 10^6)

mean(traces[:intercept])
mean(traces[:slope])


Random.seed!(0)
rwmh_traces = rwmh(model, 10^6)

mean(rwmh_traces[:intercept])
mean(rwmh_traces[:slope])

Random.seed!(0)
lmh_traces = lmh(model, 10^6,
    addr2proposal=Addr2Proposal(
        :slope => ContinuousRandomWalkProposal(1.),
        :intercept => ContinuousRandomWalkProposal(1.),
    )
)

@assert mean(lmh_traces[:intercept]) == mean(rwmh_traces[:intercept])
@assert mean(lmh_traces[:slope]) == mean(rwmh_traces[:slope])


Random.seed!(0)
hmc_traces = hmc(model, 10^4, 10, 0.1)

mean(hmc_traces[:intercept])
mean(hmc_traces[:slope])


# minimal overhead over likelihood_weighting
Random.seed!(0)
@time smc_traces, lps, marginal_lik = light_smc(model, 10^6);
W = exp.(lps);
W'smc_traces[:intercept]
W'smc_traces[:slope]

Random.seed!(0)
@time smc_traces, lps, marginal_lik = smc(model, 10^6);
W = exp.(lps);
intercept = W'smc_traces[:intercept]
slope = W'smc_traces[:slope]

X_ref = [intercept, slope]

Random.seed!(0)
@time smc_traces, lps, marginal_lik = conditional_smc(model, 1, X_ref);
W = exp.(lps);
smc_traces[:,1] == X_ref

# acceptance rate increases with more particles
Random.seed!(0)
n_particles = 100
pimh_traces = particle_IMH(model, n_particles, 1000);

mean(pimh_traces[:intercept])
mean(pimh_traces[:slope])


Random.seed!(0)
vi_result_meanfield = advi_meanfield(model, 10_000, 10, 0.01)
maximum(abs, map_mu .- vi_result_meanfield.Q.mu)
maximum(abs, map_sigma .- vi_result_meanfield.Q.sigma)

posterior = sample_posterior(vi_result_meanfield, 1_000_000)
mean(posterior[:intercept]), mean(posterior[:slope])
vi_result_meanfield.Q.mu

Random.seed!(0)
vi_result_meanfield_2 = advi(model, 10_000, 10, 0.01, MeanFieldGaussian(model.n_latents), RelativeEntropyELBO())
@assert all(vi_result_meanfield.Q.mu .≈ vi_result_meanfield_2.Q.mu)
@assert all(vi_result_meanfield.Q.sigma .≈ vi_result_meanfield_2.Q.sigma)


# ReinforceELBO
Random.seed!(0)
vi_result_bbvi = bbvi(model, 10_000, 10, 0.01)
mu_bbvi, sigma_bbvi = get_meanfield_parameters(vi_result_bbvi)
maximum(abs, map_mu .- mu_bbvi)
maximum(abs, map_sigma .- sigma_bbvi)

Random.seed!(0)
vi_result_bbvi_naive = bbvi_naive(model, 10_000, 10, 0.01)
mu_bbvi_naive, sigma_bbvi_naive = get_meanfield_parameters(vi_result_bbvi_naive)
# TODO: find out why these are not equivalent
maximum(abs, mu_bbvi .- mu_bbvi_naive)
maximum(abs, sigma_bbvi .- sigma_bbvi_naive)

Random.seed!(0)
vi_result_bbvi_rao = bbvi_rao(model, 10_000, 10, 0.01)
mu_pd_bbvi_rao, sigma_pd_bbvi_rao = get_meanfield_parameters(vi_result_bbvi_rao)
maximum(abs, map_mu .- mu_pd_bbvi_rao)
maximum(abs, map_sigma .- sigma_pd_bbvi_rao)




Random.seed!(0)
vi_result_fullrank = advi_fullrank(model, 10_000, 10, 0.01)
maximum(abs, map_mu .- vi_result_fullrank.Q.mu)
maximum(abs, map_Σ .- vi_result_fullrank.Q.base.Σ)

Random.seed!(0)
vi_result_fullrank_2 = advi(model, 10_000, 10, 0.01, FullRankGaussian(model.n_latents), RelativeEntropyELBO())
@assert all(vi_result_fullrank.Q.mu .≈ vi_result_fullrank_2.Q.base.μ)
@assert all(vi_result_fullrank.Q.base.Σ .≈ vi_result_fullrank_2.Q.base.Σ)
