# Latent Dirichlet allocation (Blei, Ng, Jordan: Latent Dirichlet Allocation, JMLR 2003).
#
# Each of the K = 2 topics is a distribution over V = 5 words, and each of the 25 documents is a mixture of topics.
# Every word in a document gets a topic z drawn from the document's topic mixture, and the word is drawn from that topic.
# We infer the topics from 262 words with single-site MCMC.
#
# Random variables in TinyPPL are univariate, so instead of Dirichlet(α, ..., α) we sample K independent Gamma(α, 1)
# variables and normalise them, which gives the same distribution.
#
# Run: julia --project=examples examples/universal/lda.jl

using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

@ppl function lda(K::Int, V::Int, doc::Vector{Int})
    M = maximum(doc)
    thetas = [] # topic distribution of each document
    for m in 1:M
        weights = [{:theta => (m, k)} ~ Gamma(1/K, 1.) for k in 1:K]
        push!(thetas, weights ./ sum(weights))
    end
    phis = [] # word distribution of each topic
    for k in 1:K
        weights = [{:phi => (k, v)} ~ Gamma(1/V, 1.) for v in 1:V]
        push!(phis, weights ./ sum(weights))
    end
    for n in eachindex(doc)
        z = {:z => n} ~ Categorical(thetas[doc[n]])
        {:w => n} ~ Categorical(phis[z])
    end
end

# word n is w[n] and belongs to document doc[n]
w = [4, 3, 5, 4, 3, 3, 3, 3, 3, 4, 5, 3, 4, 4, 5, 3, 4, 4, 4, 3, 5, 4, 5, 2, 3, 3, 1, 5, 5, 1, 4, 3, 1, 2, 5, 4, 4, 3, 5, 4, 2, 4, 5, 3, 4, 1, 4, 4, 3, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 1, 2, 2, 4, 4, 5, 4, 5, 5, 4, 3, 5, 4, 4, 4, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 3, 2, 3, 3, 5, 4, 5, 4, 3, 5, 4, 2, 2, 2, 1, 3, 2, 1, 3, 1, 3, 1, 1, 2, 1, 2, 2, 4, 4, 4, 5, 5, 4, 4, 5, 4, 3, 3, 3, 1, 3, 3, 4, 2, 1, 3, 4, 4, 5, 4, 4, 4, 3, 4, 3, 4, 5, 1, 2, 1, 3, 2, 1, 1, 2, 3, 3, 3, 3, 4, 1, 4, 4, 4, 4, 3, 4, 4, 1, 2, 2, 3, 3, 1, 1, 4, 1, 3, 1, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 5, 3, 4, 3, 1, 5, 5, 5, 3, 3, 4, 5, 3, 3, 3, 2, 3, 1, 3, 3, 1, 3, 1, 5, 5, 5, 2, 2, 3, 3, 3, 1, 1, 5, 5, 5, 3, 1, 5, 4, 1, 3, 3, 3, 3, 4, 2, 5, 1, 3, 5, 2, 5, 5, 2, 1, 3, 3, 5, 3, 5, 3, 3, 5, 1, 2, 2, 1, 1, 2, 1, 2, 3, 1, 1]
doc = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

K, V = 2, 5
args = (K, V, doc)
observations = Observations((:w => n) => w[n] for n in eachindex(w))

# Posterior mean of the word distribution of each topic, discarding the first half of the chain as burn-in.
function topic_word_distributions(traces)
    n = length(traces)
    phis = zeros(K, V)
    for trace in traces.data[n÷2+1:end], k in 1:K
        weights = [trace[:phi => (k, v)] for v in 1:V]
        phis[k, :] .+= weights ./ sum(weights)
    end
    return phis ./ (n - n÷2)
end

# The posterior is symmetric under relabelling the topics. To switch labels, a chain would have to change all topic
# assignments at once, which single-site updates practically never do, so chains with different seeds can end up with
# different labellings. Both find the same topics:
# one over the words 1, 2, 3 and one over the words 3, 4, 5.
for seed in [0, 1]
    Random.seed!(seed)
    traces = lmh(lda, args, observations, 100_000)
    phis = topic_word_distributions(traces)
    println("LMH with seed ", seed, ": P(word | topic) ≈")
    for k in 1:K
        println("  topic ", k, ": ", round.(phis[k, :], digits=2))
    end
end
