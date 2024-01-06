# ===== helpers ===========================================================================

function get_w(phi)
    k = length(phi)+1
    w = Vector{Real}(undef,k)
    s = 0 # sum_{i=1}^{j-1} w[i]
    for j in eachindex(phi)
        w[j] = (1-s) * phi[j]
        s += w[j]
    end
    w[k] = 1 - s
    return w
end

function get_phi(w)
    k = length(w)
    phi = Vector{Real}(undef,k-1)
    s = 0. # sum_{i=1}^{j-1} x[i]
    for j in 1:k-1
        phi[j] = w[j] / (1-s)
        s += w[j]
    end
    return phi
end

function get_merge_params(w1, μ1, σ1², w2, μ2, σ2²)
    # w1, w2 can be unscaled
    # w1/sum(w) + w2/sum(w) = (w1 + w1) / sum(w)
    # w1 / (w1 + w2) = (w1 / sum(w)) / ((w1 + w1) / sum(w))
    w = w1 + w2
    μ = (w1*μ1 + w2*μ2) / w
    σ² = -μ^2 + (w1*(μ1^2 + σ1²) + w2*(μ2^2 + σ2²)) / w
    return w, μ, σ²
end

function get_split_params(w, μ, σ², u1, u2, u3)
    # w can be unscaled
    # w2 / (u1 * w1) = (w2/sum(w)) / (u1 * w1/sum(w))
    # w / (u1 * w1) = (w/sum(w)) / (u1 * w1/sum(w))
    w1, w2 = w * u1, w * (1 - u1)
    μ1, μ2 = μ - u2 * sqrt(σ² * w2/w1), μ + u2 * sqrt(σ² * w1/w2)
    σ1², σ2² = u3 * (1 - u2^2) * σ² * w/w1, (1 - u3) * (1 - u2^2) * σ² * w/w2
    return w1, w2, μ1, μ2, σ1², σ2²
end

function reverse_split_params(w, μ, σ², w1, μ1, σ1², w2, μ2, σ2²)
    # w can be unscaled
    # w1/sum(w) / w/sum(w) = w1/w
    # w2/sum(w) / w1/sum(w) = w2/w1
    u1 = w1/w
    u2 = (μ - μ1) / sqrt(σ² * w2/w1)
    u3 = σ1²/σ² * u1 / (1 - u2^2)
    return u1, u2, u3
end

function split_idx(j, j_star, j1, j2)
    j == j_star ? throw(ArgumentError("component $j will be removed")) : 0
    shift1 = -(j > j_star)
    shift2 = (j + shift1) >= min(j1, j2)
    shift3 = (j + shift1 + shift2) >= max(j1, j2)
    return j + shift1 + shift2 + shift3
end

function merge_idx(j, j_star, j1, j2)
    j in (j1, j2) ? throw(ArgumentError("component $j will be removed")) : 0
    shift1 = -(j > min(j1, j2))
    shift2 = -(j > max(j1, j2))
    shift3 = (j + shift1 + shift2) >= j_star
    return j + shift1 + shift2 + shift3
end