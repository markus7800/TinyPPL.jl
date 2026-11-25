include("plotting.jl")

# Plots the clusters of a GMM trace (addresses :k, :w=>j, :μ=>j, :σ²=>j, :z=>i) against the data `gt_ys`.
# Ground-truth components are drawn in gray.
function visualize_trace(tr; raw=false)
    gaussian_pdf(μ, σ², w) = x -> w * exp(logpdf(Normal(μ, sqrt(σ²)), x))

    n = length(gt_ys)
    k = Int(tr[:k])
    colors = palette(:Paired_12)

    p = plot(xlabel="x", ylabel="density", legend=false)
    for j in 1:k
        y_js = [gt_ys[i] for i in 1:n if tr[:z=>i] == j]
        μ, σ², w = tr[:μ=>j], tr[:σ²=>j], tr[:w=>j]
        light, dark = colors[mod1(2j-1, 12)], colors[mod1(2j, 12)]

        if !isempty(y_js)
            histogram!(p, y_js, normalize=:pdf, bins=6, color=light, alpha=0.5, linewidth=0)
            scatter!(p, y_js, fill(5e-3, length(y_js)), color=dark, markersize=2, markerstrokewidth=0)
        end

        dom = (μ - 3sqrt(σ²)):1e-1:(μ + 3sqrt(σ²))
        plot!(p, dom, gaussian_pdf(μ, σ², w).(dom), color=dark)

        if j <= gt_k
            dom = (gt_μs[j] - 3sqrt(gt_σ²s[j])):1e-1:(gt_μs[j] + 3sqrt(gt_σ²s[j]))
            plot!(p, dom, gaussian_pdf(gt_μs[j], gt_σ²s[j], gt_ws[j]).(dom), color=:gray)
        end
    end
    if raw
        scatter!(p, gt_ys, fill(0.25, n), color=:black, alpha=0.5)
    end
    return p
end

plot_lps(lps) = plot(lps, ylabel="log probability", legend=false)

plot_params(params) = plot(permutedims(params), legend=false)
