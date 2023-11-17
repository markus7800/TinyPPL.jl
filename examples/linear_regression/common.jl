
import DelimitedFiles

function generate_data(slope, intercept, N, sigma)
    mkpath("data")
    xs = collect(1.:N)
    ys = round.(slope .* xs .+ intercept .+ sigma .* randn(N), digits=2)
    open("data/xs.txt", "w") do f
        DelimitedFiles.writedlm(f, xs)
    end
    open("data/ys.txt", "w") do f
        DelimitedFiles.writedlm(f, ys)
    end
    open("data/N.txt", "w") do f
        DelimitedFiles.writedlm(f, N)
    end
end


# Random.seed!(0)
# generate_data(2., 1., 100, 1.)

const slope_true = 2.
const intercept_true = -1.

const xs = vec(DelimitedFiles.readdlm(Base.pwd() * "/examples/linear_regression/data/xs.txt"))
const ys = vec(DelimitedFiles.readdlm(Base.pwd() * "/examples/linear_regression/data/ys.txt"))

#const xs = [1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799, -0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ]
#const ys = [ 2.81619183,  0.08840156,  1.24556311,  3.76987354,  3.02320312, -2.66646862,  1.18826398, -1.01462727, -0.91835056,  0.10928415]
const N = length(xs)

const slope_prior_mean = 0
const slope_prior_sigma = 3
const intercept_prior_mean = 0
const intercept_prior_sigma = 3

const σ = 2.0
m0 = [0., 0.]
S0 = [intercept_prior_sigma^2 0.; 0. slope_prior_sigma^2]
Phi = hcat(fill(1., length(xs)), xs)
const S = inv(inv(S0) + Phi'Phi / σ^2)
const map = S*(inv(S0) * m0 + Phi'ys / σ^2)