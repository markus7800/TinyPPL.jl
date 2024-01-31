using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
using Plots

@ppl function nunif()
    n ~ Geometric(0.3)
    x = 0.
    for i in 0:Int(n)
        x = {(:x,i)} ~ Uniform(x-1,x+1)
    end
end

Random.seed!(0)
vi_result = bbvi(nunif, (), Observations(),  10_000, 100, 0.01)
vi_result.Q[:n]

# TODO: fix UniversalMeanField samples values that lead to new addresses
theta, lps = sample_posterior(vi_result, 100_000);

histogram(theta[:n], normalize=true, legend=false)

histogram(theta[(:x,1)], normalize=true, legend=false)

addr = (:x,10)
histogram(theta[addr], normalize=true, legend=false)
mean(.!ismissing.(theta[addr]))

# equivalent to
# histogram(theta[addr][.!ismissing.(theta[addr])], normalize=true, legend=false)
