
using TinyPPL.Graph
using BenchmarkTools

const P = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
    0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
    0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
    0.00978, 0.02360, 0.00150, 0.01974, 0.00075]
const K = 4
const msg = [8,5,12,12,15, 23,15,18,12,4, 9, 1,13, 13,1,18,11,21,19]
# const msg = rand(1:26, 100)
const Os = ((msg .+ K) .% 26 .+ 1)[1:5]

function get_model()
    @ppl Caesar begin
        let Os = $(Main.Os),
            n = length(Os),
            K ~ DiscreteUniform(1, 26),
            P = $(Main.P),
            Fs = [{:F=>i} ~ Bernoulli(0.0001) for i in 1:n], # failed encryption
            Cs = [{:C=>i} ~ Categorical(P) for i in 1:n]
    
            [{:O=>i} ~ ((Fs[i] == 1.) ? Dirac(Cs[i] == Os[i]) : Dirac(((Cs[i] + K) % 26 + 1) == Os[i])) ↦ 1. for i in 1:n]
            # [{:O=>i} ~ Dirac(((Cs[i] + K) % 26 + 1) == Os[i]) ↦ 1. for i in 1:n]
    
            K
        end
    end
end

function print_reference_solution()
    p_f = 0.0001
    W = zeros(26)
    for k in 1:26
        W[k] += log(1/26)
        for i in 1:length(Os)
            decrypted = (Os[i] - k + 26 - 1) % 26
            decrypted = decrypted == 0 ? 26 : decrypted
            W[k] += log(p_f * P[Os[i]] + (1-p_f)*P[decrypted])
        end
    end
    W = exp.(W) / sum(exp, W)
    println("Reference: P($K)=", W[K])
end
