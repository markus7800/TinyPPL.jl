
using TinyPPL.Distributions
using TinyPPL.Graph
import Random
import Distributions: mean, var, std, MvNormal
import LinearAlgebra: diag

include("lgss/lgss_data.jl")
include("lgss/posterior.jl")

# T = 3
# y = y[1:T]
# x_gt = x_gt[1:T]

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
@time smc_traces, lps, marginal_lik = conditional_smc(LGSS, 10^3, X_ref, ancestral_sampling=true);
# 2.487067 seconds (12.79 M allocations: 2.755 GiB, 9.84% gc time)


using Plots
plot(y, label="observed");
plot!(x_gt, label="ground truth")

n_particles = 5
n_samples = 1000
Random.seed!(0)
@time traces = particle_gibbs(LGSS, n_particles, n_samples; ancestral_sampling=true, init=:zeros);

plot(1:T, fill((n_particles-1) / n_particles, T), ylims=(0,1), lc=1, legend=false);
plot!(get_update_freq(traces, n_samples), lc=1)

# visualise true posterior
mu_post, Σ_post = get_true_posterior(T, y)

inferred_mean = mean(traces.data, dims=2)
inferred_std = std(traces.data, dims=2)

# even 5 particels approximate posterior well
plot(inferred_mean, ribbon=2*inferred_std, label="estimated posterior",fillalpha=0.3);
plot!(mu_post, ribbon=2*sqrt.(diag(Σ_post)), label="true posterior", fillalpha=0.3)
plot!(x_gt, label="ground truth")
# plot!(y, label="observed")


function plot_results(Ns; n_samples, ancestral_sampling, addr2proposal=nothing)
    if addr2proposal === nothing
        addr2proposal = Addr2Proposal()
        r = "prior"
    else
        r = "posterior"
    end
    p = plot(legend=:bottomleft, title="PGM PGAS r=$r ancestral_sampling=$ancestral_sampling");
    for (i, N) in enumerate(Ns)
        println("N = $N")
        Random.seed!(0)
        @time traces = particle_gibbs(LGSS, N, n_samples; ancestral_sampling=ancestral_sampling, init=:zeros, addr2proposal=addr2proposal)
        plot!(1:T, fill((N-1)/N, T), ylims=(0,1), lc=i, label=false);
        plot!(get_update_freq(traces, n_samples), lc=i, label="N = $N")
    end
    display(p)
end

plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=false)

plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=true)

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
n_samples = 1000
Random.seed!(0)
traces = particle_gibbs(LGSS, n_particles, n_samples; ancestral_sampling=false, addr2proposal=addr2proposal);

plot(1:T, fill((n_particles-1) / n_particles, T), ylims=(0,1));
plot!(get_update_freq(traces, n_samples))


plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=false, addr2proposal=addr2proposal)
plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=true, addr2proposal=addr2proposal)


mu, Σ = get_joint_normal(T)

# compute conditional x_t | x_1:t-1, y_1:t
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

# verify that x_t | x_1:t-1, y_1:t == x_t | x_t-1, y_t
mu_cond = a*x[t-1] + σ_v^2 / (σ_e^2+σ_v^2) * (y[t] - a*x[t-1])
sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))
conditional = Normal(mu_cond, sigma_cond)

# why we cannot expect update_freq[T] == N-1 / N for ancestral_sampling = false
# weights W(x_1:t) would have to be equal (independent of x_1:t)
# but they are not

# W(x_1:t) = p(x_1:t, y_1:t) / p(x_1:t-1, y_1:t-1) r(x_t, x_1:t-1)
# = p(x_t,y_t|x_1:t-1, y_1:t-1) / r(x_t, x_1:t-1)
# = p(x_t, y_t | x_t-1) / r(x_t, x_1:t-1)

# for r(x_t, x_1:t-1) p(x_t|y_t,y_t-1)
# W(x_1:t) = p(y_t|x_t-1) - depends on x_t-1


# direct implementation of PGAS for SSM (LGSS)
import Distributions: Normal, Categorical, logpdf, mean, std, var
import Random
using Plots
import ProgressLogging: @progress

function normalise(logprobs::Vector{Float64})
    m = maximum(logprobs)
    l = m + log(sum(exp, logprobs .- m))
    return logprobs .- l
end

function f_init()
    σ_v = 0.32
    return Normal(0, σ_v)
end
function f(x_t)
    a = 0.9
    σ_v = 0.32
    return Normal(a * x_t, σ_v)
end
function g(x_t)
    σ_e = 1.
    return Normal(x_t, σ_e)
end
function r(y_t, x_t_minus_1=0; propose_from_posterior=false)
    a = 0.9
    σ_v = 0.32
    σ_e = 1.
    if !propose_from_posterior 
        return Normal(a * x_t_minus_1, σ_v) # prior
    else
        # p(x_t | y_t, x_t-1)
        mu_cond = a*x_t_minus_1 + σ_v^2 / (σ_e^2+σ_v^2) * (y_t - a*x_t_minus_1)
        sigma_cond = sqrt(σ_v^2 - (σ_v^2)^2 / (σ_e^2+σ_v^2))
        return Normal(mu_cond, sigma_cond)
    end
end

function PGAS_SSM_kernel(x_ref::Vector{Float64}, y::Vector{Float64},
    T::Int, N::Int; ancestral_sampling::Bool=false, propose_from_posterior=false)

    particles = Array{Float64}(undef, T, N)
    log_w = Vector{Float64}(undef,N)
    log_w_tilde = Vector{Float64}(undef,N)
    # t = 1
    particles[1,1] = x_ref[1]
    for i in 2:N
        q = r(y[1], propose_from_posterior=propose_from_posterior)
        x_i_1 = rand(q)
        particles[1,i] = x_i_1
    end
    for i in 1:N
        x_i_1 = particles[1,i]
        q = r(y[1], propose_from_posterior=propose_from_posterior)
        log_w[i] = logpdf(g(x_i_1), y[1]) + logpdf(f_init(), x_i_1) - logpdf(q, x_i_1)
    end

    for t in 2:T
        A = rand(Categorical(exp.(normalise(log_w))), N)
        if ancestral_sampling
            for i in 1:N
                x_i_t_minus_1 = particles[t-1,i]
                log_w_tilde[i] = log_w[i] + logpdf(f(x_i_t_minus_1), x_ref[t])
            end
            J = rand(Categorical(exp.(normalise(log_w_tilde))))
            A[1] = J
        else
            A[1] = 1
        end

        particles = particles[:,A]

        particles[t,1] = x_ref[t]
        for i in 2:N
            x_i_t_minus_1 = particles[t-1,i]
            q = r(y[t], x_i_t_minus_1, propose_from_posterior=propose_from_posterior)
            x_i_t = rand(q)
            particles[t,i] = x_i_t
        end
        for i in 1:N
            x_i_t = particles[t,i]
            x_i_t_minus_1 = particles[t-1,i]
            q = r(y[t], x_i_t_minus_1, propose_from_posterior=propose_from_posterior)
            log_w[i] = logpdf(g(x_i_t), y[t]) + logpdf(f(x_i_t_minus_1), x_i_t) - logpdf(q, x_i_t)
        end
    end

    k = rand(Categorical(exp.(normalise(log_w))))
    return particles[:,k]
end

function PGAS_SSM(y::Vector{Float64}, T::Int, n_particles::Int, n_samples::Int; ancestral_sampling::Bool=false, propose_from_posterior::Bool=false)
    X = Array{Float64}(undef, T, n_samples)

    X_current = zeros(T)
    @progress for i in 1:n_samples
        X_current = PGAS_SSM_kernel(X_current, y, T, n_particles, ancestral_sampling=ancestral_sampling, propose_from_posterior=propose_from_posterior)
        X[:,i] = X_current
    end
    return X
end


include("lgss/lgss_data.jl")
get_update_freq(traces, n_samples) = [mean(traces[t, i] != traces[t, i+1] for i in 1:n_samples-1) for t in 1:T]


function plot_ssm_results(Ns; n_samples, ancestral_sampling, propose_from_posterior)
    if propose_from_posterior
        s = "posterior"
    else
        s = "prior"
    end
    p = plot(legend=:bottomleft, title="SSM PGAS r=$s ancestral_sampling=$ancestral_sampling");
    for (i, N) in enumerate(Ns)
        println("N = $N")
        Random.seed!(0)
        @time traces = PGAS_SSM(y, T, N, n_samples, ancestral_sampling=ancestral_sampling, propose_from_posterior=propose_from_posterior)
        plot!(1:T, fill((N-1)/N, T), ylims=(0,1), lc=i, label=false);
        plot!(get_update_freq(traces, n_samples), lc=i, label="N = $N")
    end
    display(p)
end

n_particles = 5
n_samples = 1000
Random.seed!(0)
@time traces = PGAS_SSM(y, T, n_particles, n_samples, ancestral_sampling=true);

plot(1:T, fill((n_particles-1) / n_particles, T), ylims=(0,1), lc=1, legend=false);
plot!(get_update_freq(traces, n_samples), lc=1)

inferred_mean = mean(traces, dims=2)
inferred_std = std(traces, dims=2)

plot(inferred_mean, ribbon=2*inferred_std, label="estimated posterior",fillalpha=0.3);
plot!(mu_post, ribbon=2*sqrt.(diag(Σ_post)), label="true posterior", fillalpha=0.3)
plot!(x_gt, label="ground truth")
plot!(y, label="observed")


plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=false)
plot_ssm_results([5,10,100], n_samples=1000, ancestral_sampling=false, propose_from_posterior=false)

plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=true)
plot_ssm_results([5,10,100], n_samples=1000, ancestral_sampling=true, propose_from_posterior=false)

plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=false, addr2proposal=addr2proposal)
plot_ssm_results([5,10,100], n_samples = 1000, ancestral_sampling=false, propose_from_posterior=true)

plot_results([5, 10, 100], n_samples=1000, ancestral_sampling=true, addr2proposal=addr2proposal)
plot_ssm_results([5,10,100], n_samples = 1000, ancestral_sampling=true, propose_from_posterior=true)



plot_ssm_results([5,10,100,500,1000], n_samples=1000, ancestral_sampling=false, propose_from_posterior=false)
plot_ssm_results([5,10,100,500,1000], n_samples=1000, ancestral_sampling=true, propose_from_posterior=false)
