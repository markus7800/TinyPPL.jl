
using TinyPPL.Distributions
using TinyPPL.Graph
import Random

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
traces, lps = likelihood_weighting(model, 10^6)
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
