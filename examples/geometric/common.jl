
using TinyPPL.Distributions

function get_true_posterior(p, n, X)
    P_X = sum(exp(logpdf(Geometric(p), i) + logpdf(Normal(i, 1.0), X)) for i in 0:250);
    P_true = [exp(logpdf(Geometric(p), i) + logpdf(Normal(i, 1.0), X)) / P_X for i in 0:n]
    return P_true
end