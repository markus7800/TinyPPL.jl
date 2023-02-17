
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

const xs = vec(DelimitedFiles.readdlm("data/xs.txt"))
const ys = vec(DelimitedFiles.readdlm("data/ys.txt"))
const N = length(xs)

σ = 1.0
m0 = [0., 0.]
S0 = [10.0^2 0.; 0. 10.0^2]
Phi = hcat(fill(1., length(xs)), xs)
S = inv(inv(S0) + Phi'Phi / σ^2)
const map = S*(inv(S0) * m0 + Phi'ys / σ^2)