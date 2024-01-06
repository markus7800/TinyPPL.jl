using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

begin

function get_meanfield_parameters(vi_result)
    mu = [vi_result.Q[:intercept].base.μ, vi_result.Q[:slope].base.μ]
    sigma = [vi_result.Q[:intercept].base.σ, vi_result.Q[:slope].base.σ]
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

@ppl function LinReg(xs)
    intercept = {:intercept} ~ Normal(intercept_prior_mean, intercept_prior_sigma)
    slope = {:slope} ~ Normal(slope_prior_mean, slope_prior_sigma)

    for i in eachindex(xs)
        {(:y, i)} ~ Normal(f(slope, intercept, xs[i]), σ)
    end
end

args = (xs,)
observations = Observations((:y, i) => y for (i, y) in enumerate(ys));

Random.seed!(0)
vi_result_meanfield = advi_meanfield(LinReg, args, observations,  10_000, 10, 0.01)
mu_meanfield, sigma_meanfield = get_meanfield_parameters(vi_result_meanfield)
maximum(abs, map_mu .- mu_meanfield)
maximum(abs, map_sigma .- sigma_meanfield)


@ppl function LinRegGuide()
    mu1 = param("mu_intercept")
    mu2 = param("mu_slope")
    sigma1 = param("sigma_intercept", constraint=Positive())
    sigma2 = param("sigma_slope", constraint=Positive())

    {:intercept} ~ Normal(mu1, sigma1)
    {:slope} ~ Normal(mu2, sigma2)
end


Random.seed!(0)
vi_result_guide = advi(LinReg, args, observations,  10_000, 10, 0.01, LinRegGuide, (), MonteCarloELBO())


Random.seed!(0)
vi_result_bbvi = bbvi(LinReg, args, observations,  10_000, 100, 0.01)
mu_bbvi, sigma_bbvi = get_meanfield_parameters(vi_result_bbvi)

Random.seed!(0)
vi_result_reinforce = advi(LinReg, args, observations,  10_000, 100, 0.01, LinRegGuide, (), ReinforceELBO())
mu_reinforce, sigma_reinforce = get_guide_parameters(vi_result_reinforce)
# is equivalent
maximum(abs, mu_bbvi .- mu_reinforce)
maximum(abs, sigma_bbvi .- sigma_reinforce)


Random.seed!(0)
vi_result_pd = advi(LinReg, args, observations,  10_000, 10, 0.01, LinRegGuide, (), PathDerivativeELBO())
mu_pd, sigma_pd = get_guide_parameters(vi_result_pd)
maximum(abs, map_mu .- mu_pd)
maximum(abs, map_sigma .- sigma_pd)
