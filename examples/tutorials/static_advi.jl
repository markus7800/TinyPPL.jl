using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random


begin
function get_meanfield_parameters(vi_result)
    mu = [d.base.μ for d in vi_result.Q.dists]
    sigma = [d.base.σ for d in vi_result.Q.dists]
    return mu, sigma    
end

function get_guide_parameters(vi_result)
    p = get_constrained_parameters(vi_result.Q)
    mu = [p["mu_intercept"], p["mu_slope"]]
    sigma = [p["sigma_intercept"], p["sigma_slope"]]
    return mu, sigma    
end

    # xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
    xs = [-1., -0.5, 0.0, 0.5, 1.0];
    ys = [-3.2, -1.8, -0.5, -0.2, 1.5];


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

function f(slope, intercept, x)
    intercept + slope * x
end

@ppl static function LinRegStatic(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end
end

args = (xs,)
observations = Observations((:y, i) => y for (i, y) in enumerate(ys));
K = get_number_of_latent_variables(LinRegStatic, args, observations)

# ====== Meanfield ======

Random.seed!(0)
vi_result_meanfield = advi_meanfield(LinRegStatic, (xs,), observations, 10_000, 10, 0.01);
maximum(abs, map_mu .- vi_result_meanfield.Q.mu)
maximum(abs, map_sigma .- vi_result_meanfield.Q.sigma)

posterior = sample_posterior(vi_result_meanfield, 1_000_000)
mean(posterior[:intercept]), mean(posterior[:slope])
vi_result_meanfield.Q.mu

Random.seed!(0)
vi_result_meanfield_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), RelativeEntropyELBO())
# is equivalent to advi_meanfield
@assert all(vi_result_meanfield.Q.mu .≈ vi_result_meanfield_2.Q.mu)
@assert all(vi_result_meanfield.Q.sigma .≈ vi_result_meanfield_2.Q.sigma)

Random.seed!(0)
vi_result_meanfield_3 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanField([VariationalNormal() for _ in 1:K]), MonteCarloELBO())
mu_meanfield_3, sigma_meanfield_3 = get_meanfield_parameters(vi_result_meanfield_3)
# is equivalent to advi_meanfield
@assert all(vi_result_meanfield.Q.mu .≈ mu_meanfield_3)
@assert all(vi_result_meanfield.Q.sigma .≈ sigma_meanfield_3)

# PathDerivativeELBO works best, closes approximation
Random.seed!(0)
vi_result_pd = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanFieldGaussian(K), PathDerivativeELBO())
maximum(abs, map_mu .- vi_result_pd.Q.mu)
maximum(abs, map_sigma .- vi_result_pd.Q.sigma)

Random.seed!(0)
vi_result_pd_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, MeanField([VariationalNormal() for _ in 1:K]), PathDerivativeELBO())
mu_pd_2, sigma_pd_2 = get_meanfield_parameters(vi_result_pd_2)
# is equivalent to MeanFieldGaussian(K), PathDerivativeELBO()
@assert all(vi_result_pd.Q.mu .≈ mu_pd_2)
@assert all(vi_result_pd.Q.sigma .≈ sigma_pd_2)

# ReinforceELBO
Random.seed!(0)
vi_result_bbvi = bbvi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01)
mu_pd_bbvi, sigma_pd_bbvi = get_meanfield_parameters(vi_result_pd_2)


@ppl static function LinRegGuideStatic()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = param("sigma_intercept", constraint=Positive())
    # equivalent to sigma_1 = exp(param("omega_intercept"))
    sigma2 = param("sigma_slope", constraint=Positive())
    # equivalent to sigma_2 = exp(param("omega_slope"))

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end

Random.seed!(0)
@time vi_result_guide = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, LinRegGuideStatic, (), MonteCarloELBO())
mu_guide, sigma_guide = get_guide_parameters(vi_result_guide)
# is equivalent to advi_meanfield
@assert all(vi_result_meanfield.Q.mu .≈ mu_guide)
@assert all(vi_result_meanfield.Q.sigma .≈ sigma_guide)


Random.seed!(0)
@time vi_result_guide_pd = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, LinRegGuideStatic, (), PathDerivativeELBO())
mu_guide_pd, sigma_guide_pd = get_guide_parameters(vi_result_guide_pd)

# TODO: figure out where we deviate numerially
@assert all(vi_result_pd.Q.mu .≈ mu_guide_pd)
@assert all(vi_result_pd.Q.sigma .≈ sigma_guide_pd)


Random.seed!(0)
@time vi_result_guide_bbvi = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, LinRegGuideStatic, (), ReinforceELBO())
mu_guide_bbvi, sigma_guide_bbiv = get_guide_parameters(vi_result_guide_pd)

# TODO: figure out where we deviate numerially
@assert all(mu_pd_bbvi .≈ mu_guide_bbvi)
@assert all(sigma_pd_bbvi .≈ sigma_guide_bbiv)

Random.seed!(0)
vi_result_pd = advi(LinRegStatic, (xs,), observations, 10, 10, 0.01, MeanFieldGaussian(K), PathDerivativeELBO())
Random.seed!(0)
@time vi_result_guide_pd = advi(LinRegStatic, (xs,), observations, 10, 10, 0.01, LinRegGuideStatic, (), PathDerivativeELBO())
mu_guide_pd, sigma_guide_pd = get_guide_parameters(vi_result_guide_pd)

maximum(abs, mu_guide_pd .- vi_result_pd.Q.mu)
maximum(abs, sigma_guide_pd .- vi_result_pd.Q.sigma)



Random.seed!(0)
vi_result_bbvi = bbvi(LinRegStatic, (xs,), observations, 2, 2, 0.01)
mu_pd_bbvi, sigma_pd_bbvi = get_meanfield_parameters(vi_result_pd_2)

Random.seed!(0)
@time vi_result_guide_bbvi = advi(LinRegStatic, (xs,), observations, 2, 2, 0.01, LinRegGuideStatic, (), ReinforceELBO())
mu_guide_bbvi, sigma_guide_bbiv = get_guide_parameters(vi_result_guide_pd)

maximum(abs, mu_pd_bbvi .- mu_guide_bbvi)
maximum(abs, sigma_pd_bbvi .- sigma_guide_bbiv)


# ====== FullRank ======

Random.seed!(0)
vi_result_fullrank = advi_fullrank(LinRegStatic, (xs,), observations, 10_000, 10, 0.01);
maximum(abs, map_mu .- vi_result_fullrank.Q.mu)
maximum(abs, map_Σ .- vi_result_fullrank.Q.base.Σ)

Random.seed!(0)
vi_result_fullrank_2 = advi(LinRegStatic, (xs,), observations, 10_000, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO())
# is equivalent to advi_fullrank
@assert all(vi_result_fullrank.Q.mu .≈ vi_result_fullrank_2.Q.base.μ)
@assert all(vi_result_fullrank.Q.base.Σ .≈ vi_result_fullrank_2.Q.base.Σ)



