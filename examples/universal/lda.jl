
using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random

append = push!

# dirichlet does not work (only support univariate)
# change to same things as GMM

@ppl function lda(M::Int, N::Int, V::Int, doc::Vector{Int})
    K::Int = 2 # num topics

    thetas::Vector{Vector{Float64}} = Vector{Vector{Float64}}() # topic dist for doc m

    i::Int = 1
    while i <= M
        theta = {:theta => i} ~ Dirichlet(fill(1/K, K))
        thetas = append(thetas, theta)
        i = i + 1
    end

    phis::Vector{Vector{Float64}} = Vector{Vector{Float64}}() # word dist for topic k

    i = 1
    while i <= K
        phi = {:phi => i} ~ Dirichlet(fill(1/V,V))
        phis = append(phis, phi)
        i = i + 1
    end

    n::Int = 1
    while n <= N
        z = {:z => n} ~ Categorical(thetas[doc[n]])
        z = min(length(phis),z)
        {:w => n} ~ Categorical(phis[z])
        n = n + 1
    end
end


w::Vector{Int} = [4, 3, 5, 4, 3, 3, 3, 3, 3, 4, 5, 3, 4, 4, 5, 3, 4, 4, 4, 3, 5, 4, 5, 2, 3, 3, 1, 5, 5, 1, 4, 3, 1, 2, 5, 4, 4, 3, 5, 4, 2, 4, 5, 3, 4, 1, 4, 4, 3, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 1, 2, 2, 4, 4, 5, 4, 5, 5, 4, 3, 5, 4, 4, 4, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 3, 2, 3, 3, 5, 4, 5, 4, 3, 5, 4, 2, 2, 2, 1, 3, 2, 1, 3, 1, 3, 1, 1, 2, 1, 2, 2, 4, 4, 4, 5, 5, 4, 4, 5, 4, 3, 3, 3, 1, 3, 3, 4, 2, 1, 3, 4, 4, 5, 4, 4, 4, 3, 4, 3, 4, 5, 1, 2, 1, 3, 2, 1, 1, 2, 3, 3, 3, 3, 4, 1, 4, 4, 4, 4, 3, 4, 4, 1, 2, 2, 3, 3, 1, 1, 4, 1, 3, 1, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 5, 3, 4, 3, 1, 5, 5, 5, 3, 3, 4, 5, 3, 3, 3, 2, 3, 1, 3, 3, 1, 3, 1, 5, 5, 5, 2, 2, 3, 3, 3, 1, 1, 5, 5, 5, 3, 1, 5, 4, 1, 3, 3, 3, 3, 4, 2, 5, 1, 3, 5, 2, 5, 5, 2, 1, 3, 3, 5, 3, 5, 3, 3, 5, 1, 2, 2, 1, 1, 2, 1, 2, 3, 1, 1]
doc::Vector{Int} = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
M = 25  # num docs
N = 262 # total word instances
V = 5  # num words



args = (M,N,V,doc)
observations = Observations();
for n in eachindex(w)
    observations[:w => n] = w[n]
end

lmh(lda, args, observations, 10_000)
