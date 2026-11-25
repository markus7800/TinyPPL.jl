# ===== aux model ===========================================================================

@ppl function aux_update_means(tr::UniversalTrace, k::Int, n::Int)
    # update means
    for j=1:k
        y_js = [gt_ys[i] for i=1:n if tr[:z => i] == j]
        n_j = length(y_js)
        σ²_j = tr[:σ² => j]
        if !isempty(y_js)
            {:μ_proposed => j} ~ Normal((sum(y_js)/σ²_j + κ * ξ)/(n_j/σ²_j + κ), sqrt(1/(n_j/σ²_j + κ)))
        end
    end
end

@ppl function aux_update_vars(tr::UniversalTrace, k::Int, n::Int)
    # update vars
    for j=1:k
        y_js = [gt_ys[i] for i=1:n if tr[:z => i] == j]
        n_j = length(y_js)
        μ_j = tr[:μ => j]
        if !isempty(y_js)
            {:σ²_proposed => j} ~ InverseGamma(α + n_j/2, β + sum((y_js .- μ_j).^2)/2)
        end
    end
end


@ppl function aux_update_allocations(tr::UniversalTrace, k::Int, n::Int, w)
    # update allocations
    μs = [tr[:μ => j] for j=1:k]
    σ²s = [tr[:σ² => j] for j=1:k]
    for i=1:n
        y_i = gt_ys[i]
        p = [exp(logpdf(Normal(μ, sqrt(σ²)), y_i)) for (μ, σ²) in zip(μs, σ²s)] .* w
        {:z_proposed => i} ~ Categorical(p ./ sum(p))
    end
end 

@ppl function aux_update_W(tr::UniversalTrace, k::Int, n::Int)
    # update W
    counts = zeros(k)
    for i=1:n
        counts[tr[:z => i]] += 1
    end
    alpha = δ * ones(k) + counts
    α0 = sum(alpha)
    for j=1:k-1
        α0 -= alpha[j]
        {:phi_proposed => j} ~ Beta(alpha[j], α0)
    end
end

@ppl function aux_merge(tr::UniversalTrace, k::Int)
    # merge
    r1 ~ DiscreteUniform(1,k)
    r2 ~ DiscreteUniform(1,k-1)
    j_star ~ DiscreteUniform(1,k-1)
end

@ppl function aux_split(tr::UniversalTrace, k::Int, n::Int, w)
    # split
    j_star ~ DiscreteUniform(1,k)
    r1 ~ DiscreteUniform(1,k+1)
    r2 ~ DiscreteUniform(1,k)
    u1 ~ Beta(2,2)
    u2 ~ Beta(2,2)
    u3 ~ Beta(1,1)
    w1, w2, μ1, μ2, σ1², σ2² = get_split_params(w[j_star], tr[:μ => j_star], tr[:σ² => j_star], u1, u2, u3)
    
    for i=1:n
        y, z = gt_ys[i], tr[:z => i]
        if z == j_star
            p = w1 * exp(logpdf(Normal(μ1, sqrt(σ1²)), y))
            q = w2 * exp(logpdf(Normal(μ2, sqrt(σ2²)), y))
            {:to_first => i} ~ Bernoulli(p / (p + q))
        end
    end
end

@ppl function aux_model(tr::UniversalTrace, n::Int)
    k = tr[:k]
    phi = [tr[:phi=>j] for j in 1:(k-1)]
    w = get_w(phi)

    move ~ DiscreteUniform(1, k > 1 ? 6 : 5)
    if move == 1
        @subppl aux_update_means(tr, k, n)
    elseif move == 2
        @subppl aux_update_vars(tr, k, n)
    elseif move == 3
        @subppl aux_update_allocations(tr, k, n, w)
    elseif move == 4
        @subppl aux_update_W(tr, k, n)
    elseif move == 5
        @subppl aux_split(tr, k, n, w)
    elseif move == 6
        @subppl aux_merge(tr, k)
    else
        error("Unknown move.")
    end
    return move
end