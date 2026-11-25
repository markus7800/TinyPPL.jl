using TinyPPL.Distributions
using TinyPPL.Graph
using Plots


model = @pgm unif begin
    let x ~ Uniform(-1,1),
        y ~ Uniform(x-1,x+1),
        z ~ Uniform(y-1,y+1)

        {:a} ~ Normal(0., 1) ↦ 1
        z
    end
end


Random.seed!(0)
result = hmc(model, 100_000, 10, 0.1; unconstrained=true)
histogram(result[:y], normalize=true, legend=false)


Random.seed!(0)
vi_result_meanfield = advi_meanfield(model, 10_000, 10, 0.01)

posterior = sample_posterior(vi_result_meanfield, 1_000_000)
histogram(posterior[:y], normalize=true, legend=false)


Random.seed!(0)
vi_result_bbvi_rao = bbvi_rao(model, 10_000, 10, 0.01)

posterior = sample_posterior(vi_result_meanfield, 1_000_000)
histogram(posterior[:y], normalize=true, legend=false)
