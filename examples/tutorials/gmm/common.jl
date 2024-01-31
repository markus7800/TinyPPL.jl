
import PyPlot
function visualize_trace(tr; color_shift=0, raw=false)
    gaussian_pdf(μ, σ², w) = x -> w * exp(logpdf(Normal(μ, sqrt(σ²)), x));

    n, k = length(gt_ys), gt_k
    w_sum = sum(tr[:w=>j] for j in 1:k)

    cmap = PyPlot.get_cmap("Paired")
    p = PyPlot.figure()
    
    for j=1:k
        y_js = [gt_ys[i] for i=1:n if tr[:z=>i] == j]
        μ, σ² = tr[:μ=>j], tr[:σ²=>j]
        w = tr[:w=>j] / w_sum
        PyPlot.hist(y_js, density=true, bins=6, color=cmap(2j-2 + 2color_shift), alpha=0.5)
        
        dom = (μ - 3sqrt(σ²)):1e-1:μ + 3sqrt(σ²)
        PyPlot.plot(dom, gaussian_pdf(μ, σ², w).(dom), color=cmap(2j-1+ 2color_shift))
        
        dom = (gt_μs[j] - 3sqrt(gt_σ²s[j])):1e-1:gt_μs[j] + 3sqrt(gt_σ²s[j])
        PyPlot.plot(dom, gaussian_pdf(gt_μs[j], gt_σ²s[j], gt_ws[j]).(dom), color="gray")

        PyPlot.plot(y_js, 5e-3 .+ zeros(length(y_js)), ".", color=cmap(2j-1 + 2color_shift))
        PyPlot.xlabel("x"); PyPlot.ylabel("density");
    end
    if raw
        PyPlot.plot(gt_ys, 0.25 .+ zeros(n), "o", color="black", alpha=0.5)
    end
    return p
end

function plot_lps(lps)
    p = PyPlot.figure()
    PyPlot.plot(lps)
    PyPlot.ylabel("Log Probability")
    return p
end

function plot_params(params)
    p = PyPlot.figure()
    for i in 1:size(params,1)
        PyPlot.plot(params[i,:])
    end
    return p
end