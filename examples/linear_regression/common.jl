
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