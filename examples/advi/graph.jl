using TinyPPL.Distributions
using TinyPPL.Graph
using TinyPPL.Logjoint
import Random

# xs = [-1., -0.5, 0.0, 0.5, 1.0] .+ 1;
xs = [-1., -0.5, 0.0, 0.5, 1.0];
ys = [-3.2, -1.8, -0.5, -0.2, 1.5];
N = length(xs)

function f(slope, intercept, x)
    intercept + slope * x
end

slope_prior_mean = 0
slope_prior_sigma = 3
intercept_prior_mean = 0
intercept_prior_sigma = 3

σ = 2.0
m0 = [0., 0.]
S0 = [intercept_prior_sigma^2 0.; 0. slope_prior_sigma^2]
Phi = hcat(fill(1., length(xs)), xs)
S = inv(inv(S0) + Phi'Phi / σ^2) 
map_mu = S*(inv(S0) * m0 + Phi'ys / σ^2)

map_Σ = S
map_mu
map_sigma = [sqrt(S[1,1]), sqrt(S[2,2])]

model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end

    let xs = $(Main.xs),
        ys = $(Main.ys),
        slope ~ Normal($(Main.slope_prior_mean), $(Main.slope_prior_sigma)),
        intercept ~ Normal($(Main.intercept_prior_mean), $(Main.intercept_prior_sigma))

        [(Normal(f(slope, intercept, xs[i]), $(Main.σ)) ↦ ys[i]) for i in 1:$(Main.N)]
        
        (slope, intercept)
    end
end;

X = Vector{Float64}(undef, model.n_variables)
model.sample!(X)

model.logpdf(X)
model.unconstrained_logpdf!(X)


logjoint = Graph.make_logjoint(model)
K = get_number_of_latent_variables(model)


Random.seed!(0)
result = hmc_logjoint(logjoint, K, 10_000, 10, 0.1)
mean(result,dims=2)

Random.seed!(0)
result = hmc(model, 10_000, 10, 0.1)
mean(result,dims=2)


Random.seed!(0)
mu, sigma = advi_meanfield_logjoint(logjoint, K, 10_000, 10, 0.01)
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)


Random.seed!(0)
mu, sigma = advi_meanfield(model, 10_000, 10, 0.01)
maximum(abs, mu .- map_mu)
maximum(abs, sigma .- map_sigma)


Random.seed!(0)
mu, L = advi_fullrank(model, 10_000, 10, 0.01)
maximum(abs, map_mu .- mu)
maximum(abs, map_Σ .- L*L')


Random.seed!(0)
# TODO: fix input type of logjoint
Q = advi(model, 10_000, 10, 0.01, FullRankGaussian(K), RelativeEntropyELBO());
# equivalent to advi_fullrank_logjoint
maximum(abs, mu .- Q.base.μ)
maximum(abs, L*L' .- Q.base.Σ)


Random.seed!(0)
Q = bbvi(model, 10_000, 10, 0.01);
Q_mu = [d.base.μ for d in Q.dists]
Q_sigma = [d.base.σ for d in Q.dists]
maximum(abs, mu .- Q_mu)
maximum(abs, sigma .- Q_sigma)


unif = @ppl unif begin
    let x ~ Uniform(-1,1),
        y ~ Uniform(x-1,x+1),
        z ~ Uniform(y-1,y+1)

        {:a} ~ Normal(0., 1) ↦ 1
        z
    end
end
model = unif

X = Vector{Float64}(undef, model.n_variables)
model.sample!(X)
X

model.logpdf(X)

model.logpdf([0., 2., 4., 0.])
model.unconstrained_logpdf!([0., 2., 4., 0.])


Random.seed!(0)
result = hmc(model, 10_000, 10, 0.1)
histogram(result[4,:], normalize=true, legend=false)

model = @ppl (plated) plate_model begin
    let N = 3,
        x = [{:x => i} ~ Normal(0.,1.) for i in 1:N],
        a = [{:a => i} ~ Uniform(0., 1.) for i in 1:N],
        b = [{:b => i} ~ Uniform(x[i]-1,x[i]+1) for i in 1:N],
        c = [{:c => i} ~ Normal(x[i],1) for i in 1:N],
        z = [{:z => i} ~ Uniform(0.,1.) ↦ 0.1 for i in 1:N]

        N
    end
end;

X = Vector{Float64}(undef, model.n_variables);
model.sample!(X); X
Y = Vector{Float64}(undef, model.n_variables);
model.transform_to_unconstrained!(X, Y)
Z = Vector{Float64}(undef, model.n_variables);
model.transform_to_constrained!(Z, Y)
@assert all(X .≈ Z) (X,Z)

X

model.logpdf(X)

model.unconstrained_logpdf!(vcat(fill(3., 9), fill(0.5, 3)))


A = zeros(5)
B = zeros(2)

C = vcat(view(A,:),B)
C .= 1
A

result = zeros(10,100)
vcat(result, reshape(ones(2), :, 1))

