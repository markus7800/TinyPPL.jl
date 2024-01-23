
using TinyPPL.Distributions
using TinyPPL.Graph
import Random


include("lgss_data.jl")


LGSS = @pgm LGSS begin
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
end;
@assert all(get_address(LGSS, t) == (:x => t) for t in 1:T)
get_update_freq(traces, n_samples) = [sum(traces[t, i] != traces[t, i+1] for i in 1:n_samples-1)/(n_samples-1) for t in 1:T]

Random.seed!(0)
@time traces, lps = likelihood_weighting(LGSS, 10^5);
# 21.807279 seconds (240.53 M allocations: 5.098 GiB, 3.00% gc time, 0.82% compilation time)
X_ref = traces.data * exp.(lps)


Random.seed!(0)
@time smc_traces, lps, marginal_lik = light_smc(LGSS, 10^5);
# 55.446960 seconds (279.93 M allocations: 132.336 GiB, 26.71% gc time, 0.01% compilation time)

Random.seed!(0)
@time smc_traces, lps, marginal_lik = smc(LGSS, 10^5);
# 71.321295 seconds (1.08 G allocations: 147.531 GiB, 20.25% gc time, 0.04% compilation time)

Random.seed!(0)
@time smc_traces, lps, marginal_lik = conditional_smc(LGSS, 10^3, X_ref);
# 0.675084 seconds (10.19 M allocations: 1.467 GiB, 45.04% gc time, 0.37% compilation time)

Random.seed!(0)
@profview smc_traces, lps, marginal_lik = conditional_smc(LGSS, 10^3, X_ref, ancestral_sampling=true);
# 2.487067 seconds (12.79 M allocations: 2.755 GiB, 9.84% gc time)


using Plots
plot(y);
plot!(x_gt)

n_particles = 5
n_samples = 1000
Random.seed!(0)
traces = particle_gibbs(LGSS, n_particles, n_samples; ancestral_sampling=false);

plot(1:T, fill((n_particles-1) / n_particles, T), ylims=(0,1));
plot!(get_update_freq(traces, n_samples))

n_samples = 1000
p = plot(legend=:bottomleft);
for (i, N) in enumerate([5, 10, 100, 500, 1000])
    println("N = $N")
    Random.seed!(0)
    traces = particle_gibbs(LGSS, N, n_samples; ancestral_sampling=true)
    plot!(1:T, fill((N-1)/N, T), ylims=(0,1), lc=i, label=false);
    plot!(get_update_freq(traces, n_samples), lc=i, label="N = $N")
end
display(p)


import Distributions: cov, var, MvNormal

a = 0.9
σ_v = 0.32
σ_e = 1.
x = 1.
y = 1.1

x_new = rand(Normal(a*x,σ_v), 10^6)
y_new = rand.(Normal.(x_new,σ_e))
mean(x_new)
mean(y_new)
var(x_new), σ_v^2
var(y_new), σ_v^2 + σ_e^2

cov(x_new, y_new), σ_v^2
Σ = [σ_v^2 σ_v^2; σ_v^2 σ_e^2+σ_v^2]
mu = [a*x, a*x]
joint = MvNormal(mu, Σ)
marginal = Normal(mu[2], sqrt(Σ[2,2]))

mu_cond = a*x + σ_v^2 / (σ_e^2+σ_v^2) * (y - a*x)
sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))
conditional = Normal(mu_cond, sigma_cond)

ys = fill(y, length(x_new))
maximum(abs, logpdf.(conditional, x_new) - (logpdf(joint, transpose(hcat(x_new, ys))) - logpdf.(marginal, ys)))

