
using TinyPPL.Distributions
using TinyPPL.Graph
import Random


include("lgss_data.jl")
T = 3
y = y[1:T]
x_gt = x_gt[1:T]

a = 0.9
σ_v = 0.32
σ_e = 1.

LGSS = @pgm LGSS begin
    function step(t, x, a, σ_v, σ_e, y)
        let new_x = {:x => t} ~ Normal(a * x, σ_v)
            {:y => t} ~ Normal(new_x, σ_e) ↦ y[t]
            new_x
        end
    end

    let T = $(Main.T), y = $(Main.y),
        a = $(Main.a), σ_v = $(Main.σ_v), σ_e = $(Main.σ_e)

        @loop(T, step, 0, a, σ_v, σ_e, y)
    end
end;
@assert all(get_address(LGSS, t) == (:x => t) for t in 1:T)
get_update_freq(traces, n_samples) = [mean(traces[t, i] != traces[t, i+1] for i in 1:n_samples-1) for t in 1:T]

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
for (i, N) in enumerate([5, 10, 100])#, 500, 1000])
    println("N = $N")
    Random.seed!(0)
    traces = particle_gibbs(LGSS, N, n_samples; ancestral_sampling=true)
    plot!(1:T, fill((N-1)/N, T), ylims=(0,1), lc=i, label=false);
    plot!(get_update_freq(traces, n_samples), lc=i, label="N = $N")
end
display(p)

struct LGSSProposal <: Distributions.ProposalDistribution
    a::Float64
    σ_v::Float64
    σ_e::Float64
    ys::Vector{Float64}
end

function Distributions.proposal_dist(dist::LGSSProposal, x_current::Tuple{Pair,Vector{Float64}})
    addr, X = x_current
    t::Int = addr[2]
    a = dist.a
    σ_v = dist.σ_v
    σ_e = dist.σ_e

    x_last = t == 1 ? 0 : X[t-1]
    y_t = dist.ys[t] 

    mu_cond = a*x_last + (σ_v^2 / (σ_e^2+σ_v^2)) * (y_t - a*x_last)
    sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))

    return Normal(mu_cond, sigma_cond)
end

addr2proposal = Addr2Proposal((:x => t) => LGSSProposal(a, σ_v, σ_e, y) for t in 1:T);

n_particles = 5
n_samples = 2
Random.seed!(0)
traces = particle_gibbs(LGSS, n_particles, n_samples; ancestral_sampling=false, addr2proposal=addr2proposal);

plot(1:T, fill((n_particles-1) / n_particles, T), ylims=(0,1));
plot!(get_update_freq(traces, n_samples))




Random.seed!(0)
@time traces, lps = likelihood_weighting(LGSS, 10^5);
X_ref = traces.data * exp.(lps)

Random.seed!(0)
@time smc_traces, lps, marginal_lik = smc(LGSS, 3, addr2proposal=addr2proposal);

Random.seed!(0)
@time smc_traces, lps, marginal_lik = conditional_smc(LGSS, 3, X_ref, addr2proposal=addr2proposal);



import Distributions: cov, var, MvNormal
x = 1.
y_obs = 1.1

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

mu_cond = a*x + σ_v^2 / (σ_e^2+σ_v^2) * (y_obs - a*x)
sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))
conditional = Normal(mu_cond, sigma_cond)

ys = fill(y_obs, length(x_new))
maximum(abs, logpdf.(conditional, x_new) - (logpdf(joint, transpose(hcat(x_new, ys))) - logpdf.(marginal, ys)))


T = 3
# X_t = v_1 + a v_2 + a^2 v_3 ..., v_i ~ Normal(0,σ_v)
# cov(X + Y, W + V) = cov(X,W) + cov(X,V) + cov(Y,W) + cov(Y,V)
# cov(X_t, X_{t+1}) = cov(X_t,a X_t + v_{t+1}) = a cov(X_t, X_t)

# [x_1, y_1, x_2, y_2, ..., x_T, y_T]
function get_Σ(T)
    Σ = zeros(2*T, 2*T)
    for t in 1:T
        x_ix = 2*(t-1) + 1
        y_ix = x_ix + 1
        Σ[x_ix, x_ix] = σ_v^2 * ((a^2)^t - 1) / (a^2 - 1) 
        Σ[y_ix, y_ix] = Σ[x_ix, x_ix] + σ_e^2
        Σ[x_ix, y_ix] = Σ[x_ix, x_ix]

        for (n,t2) in enumerate(t+1:T)
            x_ix_2 = 2*(t2-1) + 1
            y_ix_2 = x_ix_2 + 1
            Σ[x_ix, x_ix_2] = a^n * Σ[x_ix, x_ix]

            Σ[x_ix, y_ix_2] = Σ[x_ix, x_ix_2]

            Σ[y_ix, x_ix_2] = Σ[x_ix, x_ix_2]

            Σ[y_ix, y_ix_2] = Σ[x_ix, x_ix_2]
        end
    end
    for i in 1:2*T, j in (i+1):2*T
        Σ[j,i] = Σ[i,j]
    end

    reorder = vcat([2*(t-1) + 1 for t in 1:T], [2*(t-1) + 1 for t in 1:T] .+ 1)
    # [x_1, x_2, ..., x_T, y_1, y_2, ..., y_T]
    Σ  = Σ[reorder,reorder]
    return Σ
end

mu = fill(0,2*T)
Σ = get_Σ(T)

Z = rand(MvNormal(mu, Σ))
x = Z[1:T]
y = Z[T+1:end]

t = 2
mask = vcat([t], 1:(t-1), (T+1):(T+t))
# mask = vcat([t], 1:(t-1))
Σt = Σ[mask, mask] # [x_t, x_1, x_2, ..., x_t-1, y_1, y_2, ..., y_t]
mut = mu[mask]

mu1 = mut[1:1]
mu2 = mut[2:end]

Σ11 = Σt[1:1,1:1]
Σ12 = Σt[1:1,2:end]
Σ21 = Σt[2:end,1:1]
Σ22 = Σt[2:end,2:end]

condition_on =  vcat(x[1:(t-1)], y[1:t])
# condition_on =  vcat(x[1:(t-1)])

sigma_cond = sqrt((Σ11 - (Σ12 * (Σ22 \ Σ21))))[1,1]
mu_cond = (mu1 + Σ12 * (Σ22 \ (condition_on - mu2)))[1]
Normal(mu_cond, sigma_cond)

x_last = x[t-1]
y_obs = y[t]
mu_cond = a*x_last + σ_v^2 / (σ_e^2+σ_v^2) * (y_obs - a*x_last)
sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))
conditional = Normal(mu_cond, sigma_cond)

# W(x_1:t) = p(x_1:t, y_1:t) / p(x_1:t-1, y_1:t-1) r(x_t, x_1:t-1)
# = p(x_t,y_t|x_1:t-1, y_1:t-1) / r(x_t, x_1:t-1)
# = p(x_t, y_t | x_t-1) / r(x_t, x_1:t-1)

# for r(x_t, x_1:t-1) p(x_t|y_t,y_t-1)
# W(x_1:t) = p(y_t|x_t-1) - depends on x_t-1


struct LGSSFullConditionalProposal <: Distributions.ProposalDistribution
    T::Int
    a::Float64
    σ_v::Float64
    σ_e::Float64
    ys::Vector{Float64}
    Σ::Matrix{Float64} # [x_1, x_2, ..., x_T, y_1, y_2, ..., y_T]
end

function Distributions.proposal_dist(dist::LGSSFullConditionalProposal, x_current::Tuple{Pair,Vector{Float64}})
    addr, X = x_current
    t::Int = addr[2]
    a = dist.a
    σ_v = dist.σ_v
    σ_e = dist.σ_e
    T = dist.T

    mask = vcat([t], 1:(t-1), (T+1):(T+t))
    Σt = dist.Σ[mask, mask] # [x_t, x_1, x_2, ..., x_t-1, y_1, y_2, ..., y_t]

    Σ11 = Σt[1:1,1:1]
    Σ12 = Σt[1:1,2:end]
    Σ21 = Σt[2:end,1:1]
    Σ22 = Σt[2:end,2:end]

    condition_on = vcat(dist.ys[1:t], X[1:(t-1)])
    
    sigma_cond = sqrt((Σ11 - (Σ12 * (Σ22 \ Σ21)))[1,1])
    mu_cond = (Σ12 * (Σ22 \ condition_on))[1] # mu1 0 0, mu2 = 0

    return Normal(mu_cond, sigma_cond)
end

Σ = get_Σ(T)
addr2proposal = Addr2Proposal((:x => t) => LGSSFullConditionalProposal(T, a, σ_v, σ_e, y, Σ) for t in 1:T);

Random.seed!(0)
@time smc_traces, lps, marginal_lik = smc(LGSS, 3, addr2proposal=addr2proposal);

Random.seed!(0)
@time traces, lps = likelihood_weighting(LGSS, 10^5);
X_ref = traces.data * exp.(lps)

Random.seed!(0)
@time smc_traces, lps, marginal_lik = conditional_smc(LGSS, 3, X_ref, addr2proposal=addr2proposal);
