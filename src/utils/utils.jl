import MacroTools

rmlines(expr) = MacroTools.postwalk(sub_ex -> MacroTools.rmlines(sub_ex), expr)


function normalise(logprobs::Vector{Float64})
    m = maximum(logprobs)
    l = m + log(sum(exp, logprobs .- m))
    return logprobs .- l
end