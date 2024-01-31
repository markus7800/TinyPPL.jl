
using TinyPPL.Distributions
using TinyPPL.Graph
import Random


model = @pgm simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end


begin
    xs = [-1., -0.5, 0.0, 0.5, 1.0];
    ys = [-3.2, -1.8, -0.5, -0.2, 1.5];
    N = length(xs)

    slope_prior_mean = 0
    slope_prior_sigma = 3
    intercept_prior_mean = 0
    intercept_prior_sigma = 3

    σ = 2.0
end
model = @pgm plated LinReg begin
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
# TODO: maybe data nodes + data plates?


const gt_k = 4
const gt_ys = [-7.87951290075215, -23.251364738213493, -5.34679518882793, -3.163770449770572,
10.524424782864525, 5.911987013277482, -19.228378698266436, 0.3898087330050574,
8.576922415766697, 7.727416085566447]
const gt_zs = [2, 1, 2, 2, 3, 3, 1, 2, 3, 3]
const gt_ws = [0.20096082191563705, 0.22119959941799663, 0.3382086364817468, 0.23963094218461967]
const gt_μs = [-20.0, 0.0, 10.0, 30.0]
const gt_σ²s = [3.0, 8.0, 7.0, 1.0]

model = @pgm plated GMM begin
    function dirichlet(δ, k)
        let w = [{:w=>i} ~ Gamma(δ, 1) for i in 1:k]
            w / sum(w)
        end
    end
    let λ = 3, δ = 5.0, ξ = 0.0, κ = 0.01, α = 2.0, β = 10.0,
        k = ({:k} ~ Poisson(λ) ↦ 3) + 1, # observed
        y = $(Main.gt_ys),
        n = length(y),
        w = dirichlet(δ, k),
        means = [{:μ=>j} ~ Normal(ξ, 1/sqrt(κ)) for j in 1:k],
        vars = [{:σ²=>j} ~ InverseGamma(α, β) for j in 1:k],
        z = [{:z=>i} ~ Categorical(w) for i in 1:n]

        [{:y=>i} ~ Normal(means[Int(z[i])], sqrt(vars[Int(z[i])])) ↦ y[i] for i in 1:n]
        
        means
    end
end

model = @pgm ncoins begin
    let X ~ Uniform(-1.,1.),
        Y ~ Uniform(X-1.,X+1.),
        Z ~ Uniform(Y-1.,Y+1.)
        X
    end
end


model = @pgm ncoins begin
    function f(x)
        x + 1
    end
    let X ~ Uniform(-1.,1.)
        @iterate(5, f, X)
    end
end


model = @pgm ncoins begin
    function f(count, val, x , y)
        val * x + y
    end
    let X ~ Uniform(-1.,1.)
        @loop(5, f, X, 2, 1)
    end
end

T = 400
y = randn(T)

model = @pgm lgss begin
    function step(t, x, a, σ_v, σ_e, y)
        let new_x = {:x => t} ~ Normal(a * x, σ_v)
            {:y => t} ~ Normal(new_x, σ_e) ↦ y[t]
            new_x
        end
    end

    let T = $(Main.T), y = $(Main.y),
        a = 0.9, σ_v = 0.32, σ_e = 1.

        @loop(T, step, 0, a, σ_v, σ_e, y)
    end
end



model = @pgm lazy_ifs branching_model begin
    let b ~ Bernoulli(0.5),
        mu = if b == 1.
            let x ~ Normal(-1,1)
                x
            end
        else
            let y ~ Normal(1,1)
                y
            end
        end,
        z ~ Normal(mu,1)

        z
    end
end
# b, x, y, z
model.addresses

# with lazy_ifs -3.0310242469692907
# without lazy_ifs -5104.449962780174
model.logpdf([1., 0., -100., 0.], Float64[])



model = @pgm normal begin
    let X ~ Normal(0., 1.)
        if X < 0
            let Y ~ Normal(1.,1.)
                Y
            end
        else
            let Z ~ Normal(2.,1.)
                Z
            end
        end
    end
end

f_main(x) = x^2
model = @pgm fcalls begin
    function f_pgm(x)
        x^3
    end
    let x ~ Normal(0., 1)
        (Main.f_main(x), f_pgm(x))
    end
end
# Return expression:
# (Main.f_main(x1), x1 ^ 3)

get_retval(model, [2.])