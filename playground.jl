using PPL.Distributions
using PPL.Trace

@ppl function geometric(p::Float64, observed::Bool)
    i = 0
    while true
        b = {(:b,i)} ~ Bernoulli(p)
        b && break
        i += 1
    end
    if observed
        {:X} ~ Normal(i, 1.0)
    end
    return i
end

@ppl function geometric_recursion(p::Float64, observed::Bool, i::Int)
    b = {(:b,i)} ~ Bernoulli(p)
    if b
        if observed
            {:X} ~ Normal(i, 1.0)
        end
        return i
    else
        return @subtrace geometric_recursion(p, observed, i+1)
    end
end

observations = Dict(:X => 5);
@time traces, retvals, lps = importance_sampling(geometric, (0.5, true), observations, 1_000_000);

@time traces, retvals, lps = importance_sampling(geometric_recursion, (0.5, true, 0), observations, 1_000_000);


W = exp.(lps);
P_hat = [sum(W[retvals .== i]) for i in 0:10]

P_X = sum(exp(logpdf(Geometric(0.5), i) + logpdf(Normal(i, 1.0), observations[:X])) for i in 0:100);
P_true = [exp(logpdf(Geometric(0.5), i) + logpdf(Normal(i, 1.0), observations[:X])) / P_X for i in 0:10]

maximum(abs.(P_hat .- P_true))

using PPL.Distributions
using PPL.Graph

foppl = quote
    let A ~ Bernoulli(0.5),
        B = (Bernoulli(A == 1 ? 0.2 : 0.8) ↦ false),
        C ~ Bernoulli(B == 1 ? 0.9 : 0.7),
        D = (Bernoulli(C == 1 ? 0.5 : 0.2)) ↦ true
        
        A + C
    end
end;

pgm = compile_pgm(foppl);

@time traces, retvals, lps = importance_sampling(pgm, 1_000_000);