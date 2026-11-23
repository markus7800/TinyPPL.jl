module TestUtils

export logsumexp, normalise_logw, wmean, wstd, wprob, burn, frequencies

logsumexp(x) = (m = maximum(x); m == -Inf ? -Inf : m + log(sum(exp, x .- m)))

"""
Normalises log weights (importance algorithms return normalised log weights, see REVIEW.md A1,
this also accepts unnormalised ones).
"""
normalise_logw(logw) = logw .- logsumexp(logw)

function wmean(xs, logw)
    w = exp.(normalise_logw(logw))
    return sum(w .* xs)
end

function wstd(xs, logw)
    w = exp.(normalise_logw(logw))
    m = sum(w .* xs)
    return sqrt(sum(w .* (xs .- m) .^ 2))
end

"weighted probability of `xs .== value`"
wprob(xs, logw, value) = wmean(xs .== value, logw)

"drop the first `frac` of a chain"
burn(xs, frac=0.1) = xs[floor(Int, length(xs) * frac)+1:end]

"empirical frequency of each value in `values`"
frequencies(xs, values) = [sum(xs .== v) / length(xs) for v in values]

end # module TestUtils
