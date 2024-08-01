using TinyPPL.Distributions
using TinyPPL.Graph
import TinyPPL.Graph: Graph

import Pkg; Pkg.activate("test")
import Distributions as Dists
using Plots


model = @pgm PriorMix begin
    let N = 10,
        y = [1.6024, 0.880643, -0.792556, -1.02184, -0.530578, 0.494777, -1.51075, -0.630318, -1.18208, 0.187967],
        
        b ~ Bernoulli(0.8),
        m = (2 * b - 1) * 4,
        
        # u ~ Uniform(0.,1.),
        # m = (2 * (u < 0.8) - 1) * 4,

        mu ~ Main.Dists.truncated(Normal(m,0.5),-5.,5)

        [{:y => i} ~ TransformedDistribution(TDist(5), AffineTransform(2., mu)) ↦ y[i] for i in 1:N]

        mu
    end
end

N = 40400

@time result = aqua(model, N, method=:ve);
@time result = aqua(model, N, method=:bp);
@time result = aqua(model, N, method=:jt);

xs, ps = result[:mu]
plot(xs, ps)

xs, ps = result[:u]
plot(xs, ps, ylim=(0,5))
sum(ps[xs .< 0.8]) / sum(ps) # u < 0.8 == true
sum(ps[xs .> 0.8]) / sum(ps) # u < 0.8 == false

xs, ps = result[:b]


ys = [1.6024, 0.880643, -0.792556, -1.02184, -0.530578, 0.494777, -1.51075, -0.630318, -1.18208, 0.187967]
mus = LinRange(-5,5,1000)
prior = log.([0.8 * Dists.pdf(Normal(4,0.5), mu) + 0.2 * Dists.pdf(Normal(-4,0.5), mu) for mu in mus])
lik = [(sum(logpdf(TransformedDistribution(TDist(5), AffineTransform(2., mu)), y) for y in ys)) for mu in mus]
#lik = [exp(sum(logpdf(Normal(mu,0.5),y) for y in ys)) for mu in mus]
plot(mus, exp.(prior))
plot(mus, exp.(lik))
plot(mus, exp.(prior .+ lik))

Random.seed!(0)
@time traces, lps = likelihood_weighting(model, 10^6)
histogram(traces[:mu], weights=exp.(lps), normalize=true, legend=false, lc=1)

Random.seed!(0)
@time traces = lmh(model, 10^6, addr2proposal=Addr2Proposal(:mu => StaticProposal(Uniform(-5,5))))#, :u => StaticProposal(Bernoulli(0.2))))
histogram(traces[:mu], normalize=true, legend=false, lc=1)
traces[:u]




model = @pgm ZeroOne begin
    let N = 20,
        y = [1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1],
        x1 = [6, 8, -1, 0, 5, 1.2, -2, 9.8, 4, 12, 1, 10, 1, 2.2, -6, 9.8, 1, 1, 1, 1],
        
        w1 ~ Uniform(-10,10),
        w2 ~ Uniform(-10,10)

        [Exponential(1/exp(y[i] * (x1[i] * w1 + w2) < 0 ? 0 : 1)) ↦ 0. for i in 1:N]

        w2
    end
end

f = π
logpdf(Exponential(1/exp(f)),0.)

N = 200
@time result = aqua(model, N, method=:ve);
@time result = aqua(model, N, method=:bp);
@time result = aqua(model, N, method=:jt);

xs, ps = result[:w2]
plot(xs, ps)


model = @pgm Tug begin
    let N = 40,
        y = [0,1,1,1,0,0,1,0,0,1,0,1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1],

        b_alice ~ Bernoulli(0.33),
        alice ~ Main.Dists.truncated(b_alice == 1 ? Normal(1.,1.) : Normal(0.5,0.5), -5, 5),

        b_bob ~ Bernoulli(0.33),
        bob ~ Main.Dists.truncated(b_bob == 1 ? Normal(1.,1.) : Normal(0.5,0.5), -5, 5)

        [Exponential(1/exp((abs(alice) > abs(bob)) == y[i] ? 1 : 0)) ↦ 0. for i in 1:N]

        alice
    end
end

N = 60
@time result = aqua(model, N, method=:ve);
@time result = aqua(model, N, method=:bp);
@time result = aqua(model, N, method=:jt);

xs, ps = result[:alice]
plot(xs, ps)



model = @pgm AlterMu begin
    let N = 40,
        y = [-2.57251482,  0.33806206,  2.71757796,  1.09861336,  2.85603752,
        -0.91651351,  0.15555127, -2.68160347,  2.47043789,  3.47459025,
        1.63949862, -1.32148757,  2.64187513,  0.30357848, -4.09546231,
        -1.50709863, -0.99517866, -2.0648892 , -2.40317949,  3.46383544,
        0.91173696,  1.18222221,  0.04235722, -0.52815171,  1.15551598,
        -1.62749724,  0.71473237, -1.08458812,  4.66020296,  1.24563831,
        -0.67970862,  0.93461681,  1.18187607, -1.49501051,  2.44755622,
        -2.06424237, -0.04584074,  1.93396696,  1.07685273, -0.09837907],

        mu1 ~ Main.Dists.truncated(Normal(0.,5.), -2, 2),
        mu2 ~ Main.Dists.truncated(Normal(0.,5.), -2, 2),
        mu3 ~ Main.Dists.truncated(Normal(0.,5.), -2, 2)

        [Normal((3.0 * mu1 * mu2) - mu3, 1.) ↦ y[i] for i in 1:N]
        
        mu1
    end
end


N = 60
@time result = aqua(model, N, method=:ve);
@time result = aqua(model, N, method=:bp);
@time result = aqua(model, N, method=:jt);

xs, ps = result[:mu1]
plot(xs, ps)



model = @pgm AlterMu2 begin
    let N = 40,
        y = [-2.57251482,  0.33806206,  2.71757796,  1.09861336,  2.85603752,
        -0.91651351,  0.15555127, -2.68160347,  2.47043789,  3.47459025,
        1.63949862, -1.32148757,  2.64187513,  0.30357848, -4.09546231,
        -1.50709863, -0.99517866, -2.0648892 , -2.40317949,  3.46383544,
        0.91173696,  1.18222221,  0.04235722, -0.52815171,  1.15551598,
        -1.62749724,  0.71473237, -1.08458812,  4.66020296,  1.24563831,
        -0.67970862,  0.93461681,  1.18187607, -1.49501051,  2.44755622,
        -2.06424237, -0.04584074,  1.93396696,  1.07685273, -0.09837907],

        mu1 ~ Uniform(-10,10),
        mu2 ~ Uniform(-10,10)

        [Normal(mu1 + mu2, 1.) ↦ y[i] for i in 1:N]
        
        mu1
    end
end

N = 200
@time result = aqua(model, N, method=:ve);
@time result = aqua(model, N, method=:bp);
@time result = aqua(model, N, method=:jt);

xs, ps = result[:mu1]
plot(xs, ps)


model = @pgm AnovaRP begin
    let N = 40,
        J = 1,
        # county = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # x = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        y = [0.617007250574049, 0.895262019894671, -0.729874754380836,
        0.301471152630035, 0.929871984146596, 1.70686064033886, 1.09782179678289,
        1.27082634042846, 1.75132331738128, 1.03490015378005, 1.71226194303743,
        0.432311329644381, 1.04451460234109, 0.797605506723309, 1.19837496179441,
        2.00164115453824, -0.377675819779764, -0.148489586942737, 2.12586836175586,
        0.560257670828821, 0.751966977556745, 0.346571453512205, 1.2187492301484,
        1.56886622653154, 1.38035019540838, 2.76837231107256, 0.323735749142999,
        1.42941129934677, 0.468406777114355, 0.751905484130352, 0.568364441627544,
        0.0139444580475871, -0.152744699487943, 1.89057571718701, 1.48479996615007,
        1.87341593940818, 1.89344977488441, 0.537059334676276, 0.944371774385459,
        -0.474992028001801],

        a ~ Uniform(0., 3.),
        mu1 ~ Uniform(-10,10),
        sigma_y ~ Uniform(0.,100),
        robust_local_tau = [{:tau => i} ~ Uniform(0, 10) for i in 1:N]

        [Normal(a, sigma_y / sqrt(max(1e-5,robust_local_tau[i]))) ↦ y[i] for i in 1:N]

        
        a
    end
end

N = 60
@time result = aqua(model, N, method=:ve);
@time result = aqua(model, N, method=:bp);
@time result = aqua(model, N, method=:jt);

xs, ps = result[:a]
plot(xs, ps)

Random.seed!(0)
@time traces = lmh(model, 10^6)
histogram(traces[:a], normalize=true, legend=false, lc=1,bins=100)
plot!(xs, ps)



model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Normal(X, 1.)

        Normal(Y, 1.) ↦ 3.
    end
end

result = aqua(model, 500);

xs, x_ps = result[:X]
ys, y_ps = result[:Y]
plot(xs, x_ps)
plot!(ys, y_ps)


result[:b][2]

Random.seed!(0)
@time traces, lps = likelihood_weighting(model, 10^6)
histogram(traces[:X], weights=exp.(lps), normalize=true, legend=false, lc=1)
plot!(xs, x_ps)


model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Normal(0, 1.),
        Z ~ Normal(X + Y, 1e-1)

        Z
    end
end

result = aqua(model, 100, method=:ve);

xs, x_ps = result[:X]
ys, y_ps = result[:Y]
zs, z_ps = result[:Z]
plot(xs, x_ps)
plot!(ys, y_ps)
plot!(zs, z_ps)



model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Normal(0, 1.)
        
        Normal(X + Y, 1.) ↦ 1.
    end
end

result = aqua(model, 100, method=:ve);


xs, x_ps = result[:X]
ys, y_ps = result[:Y]
plot(xs, x_ps)
plot!(ys, y_ps)

support, joint = aqua_get_joint(model, result, Address[:X,:Y])
sum(joint) * (xs[2]-xs[1]) * (ys[2]-ys[1])

heatmap(support[1], support[2], joint)


model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Normal(0, 1.)
        X + Y
    end
end

result = aqua(model, 100, method=:ve);

xs, x_ps = result[:X]
ys, y_ps = result[:Y]
plot(xs, x_ps)
plot!(ys, y_ps)

zs, zps = aqua_get_return_distribution(model, result)
plot(zs, zps)
plot!(z -> exp(logpdf(Normal(0,sqrt(2.)),z)))


model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Uniform(0, 1.)
        X + Y
    end
end
result = aqua(model, 100, method=:ve);
zs, zps = aqua_get_return_distribution(model, result)
plot(zs, zps)
plot!(z -> exp(logpdf(Normal(0.5,sqrt(1+Dists.var(Uniform(0,1)))),z)))



model = @pgm Model begin
    let X ~ Normal(0.,1.),
        Y ~ Normal(0, 1.)
        X / Y
    end
end
result = aqua(model, 100, method=:ve);
zs, zps = aqua_get_return_distribution(model, result)
plot(zs, zps)
plot!(z -> exp(logpdf(Cauchy(),z)))
