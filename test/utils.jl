module TestUtils

using Test

export logsumexp, normalise_logw, wmean, wstd, wprob, burn, frequencies, test_trace_indexing

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

# traces[addr, ixs] agrees with traces[addr][ixs] for ranges, index vectors and Bool masks,
# and traces[:, ixs] still selects a subset of the traces (no method ambiguity between the two)
function test_trace_indexing(traces, addr)
    values = traces[addr]
    @test traces[addr, 2:5] == values[2:5]
    @test traces[addr, [3, 1]] == [traces[addr, 3], traces[addr, 1]]
    mask = values .> 0
    @test traces[addr, mask] == values[mask]
    @test length(traces[:, 2:5]) == 4
    @test traces[:, 2:5][addr] == values[2:5]
end

end # module TestUtils
