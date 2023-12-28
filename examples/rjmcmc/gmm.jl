using TinyPPL.Distributions
using TinyPPL.Evaluation
import Random
import LinearAlgebra: det, logabsdet
# using Plots

const gt_k = 4
const gt_ys = [-7.87951290075215, -23.251364738213493, -5.34679518882793, -3.163770449770572,
10.524424782864525, 5.911987013277482, -19.228378698266436, 0.3898087330050574,
8.576922415766697, 7.727416085566447, -18.043123523482492, 9.108136117789305,
29.398734347901787, 2.8578485031858003, -20.716691460295685, -18.5075008084623,
-21.52338318392563, 10.062657028986715, -18.900545157827718, 3.339430437507262,
3.688098690412526, 4.209808727262307, 3.371091291010914, 30.376814419984456,
12.778653273596902, 28.063124205174137, 10.70527515161964, -18.99693615834304,
8.135342537554163, 29.720363913218446, 29.426043027354385, 28.40516772785764,
31.975585225366686, -20.642437143912638, 30.84807631345935, -21.46602061526647,
12.854676808303978, 30.685416799345685, 5.833520737134923, 7.602680172973942,
10.045516408942117, 28.62342173081479, -20.120184774438087, -18.80125468061715,
12.849708921404385, 31.342270731653656, 4.02761078481315, -19.953549865339976,
-2.574052170014683, -21.551814470820258, -2.8751904316333268,
13.159719198798443, 8.060416669497197, 12.933573330915458, 0.3325664001681059,
11.10817217269102, 28.12989207125211, 11.631846911966806, -15.90042467317705,
-0.8270272159702201, 11.535190070081708, 4.023136673956579,
-22.589713328053048, 28.378124912868305, -22.57083855780972,
29.373356677376297, 31.87675796607244, 2.14864533495531, 12.332798078071061,
8.434664672995181, 30.47732238916884, 11.199950328766784, 11.072188217008367,
29.536932243938097, 8.128833670186253, -16.33296115562885, 31.103677511944685,
-20.96644212192335, -20.280485886015406, 30.37107537844197, 10.581901339669418,
-4.6722903116912375, -20.320978011296315, 9.141857987635252, -18.6727012563551,
7.067728508554964, 5.664227155828871, 30.751158861494442, -20.198961378110013,
-4.689645356611053, 30.09552608716476, -19.31787364001907, -22.432589846769154,
-0.9580412415863696, 14.180597007125487, 4.052110659466889,
-18.978055134755582, 13.441194891615718, 7.983890038551439, 7.759003567480592]
const gt_zs = [2, 1, 2, 2, 3, 3, 1, 2, 3, 3, 1, 3, 4, 2, 1, 1, 1, 3, 1, 2, 2, 3, 2, 4, 3, 4,
      3, 1, 3, 4, 4, 4, 4, 1, 4, 1, 3, 4, 3, 3, 3, 4, 1, 1, 3, 4, 3, 1, 2, 1, 2,
      3, 3, 3, 2, 3, 4, 3, 1, 2, 3, 2, 1, 4, 1, 4, 4, 2, 3, 3, 4, 3, 3, 4, 3, 1,
      4, 1, 1, 4, 3, 2, 1, 3, 1, 3, 3, 4, 1, 2, 4, 1, 1, 2, 3, 2, 1, 3, 3, 3]
const gt_ws = [0.20096082191563705, 0.22119959941799663, 0.3382086364817468, 0.23963094218461967]
const gt_μs = [-20.0, 0.0, 10.0, 30.0]
const gt_σ²s = [3.0, 8.0, 7.0, 1.0]
const n = length(gt_ys)

const λ = 3
const δ = 5.0
const ξ = 0.0
const κ = 0.01
const α = 2.0
const β = 10.0

# does work, because no bijection from gamma to dirichlet
# @ppl function dirichlet(δ, k)
#     w = [{:w=>j} ~ Gamma(δ, 1) for j in 1:k]
#     return w / sum(w)
# end

# function get_w(tr::Dict{Any,Real})
#     w = [tr[:w => j] for j in 1:tr[:k]]
#     return w / sum(w)
# end

@ppl function dirichlet(δ, k)
    w = zeros(k)
    α0 = δ * k
    s = 0 # sum_{i=1}^{j-1} w[i]
    for j in 1:k-1
        α0 -= δ
        phi = {:phi => j} ~ Beta(δ, α0)
        w[j] = (1-s) * phi
        s += w[j]
    end
    w[k] = 1 - s
    return w
end

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


import Distributions
struct PositivePoisson <: Distributions.DiscreteUnivariateDistribution
    λ
end
Distributions.rand(d::PositivePoisson) = Distributions.rand(Poisson(d.λ)) + 1
Distributions.logpdf(d::PositivePoisson, x) = Distributions.logpdf(Poisson(d.λ), x-1)

@ppl function gmm(n::Int)
    k = {:k} ~ PositivePoisson(λ)
    w = @subppl dirichlet(δ, k)

    means, vars = zeros(k), zeros(k)
    for j=1:k
        means[j] = ({:μ=>j} ~ Normal(ξ, 1/sqrt(κ)))
        vars[j] = ({:σ²=>j} ~ InverseGamma(α, β))
    end
    for i=1:n
        z = {:z=>i} ~ Categorical(w)
        {:y=>i} ~ Normal(means[z], sqrt(vars[z]))
    end
end

const observations = Dict{Any, Real}((:y=>i)=>y for (i, y) in enumerate(gt_ys))
# observations[:k] = gt_k-1

@ppl function aux_model(tr::Dict{Any,Real}, n::Int)
    k = tr[:k]
    phi = [tr[:phi=>j] for j in 1:(k-1)]
    w = get_w(phi)

    move ~ DiscreteUniform(1,6)
    if move == 1
        # update means
        for j=1:k
            y_js = [gt_ys[i] for i=1:n if tr[:z => i] == j]
            n_j = length(y_js)
            σ²_j = tr[:σ² => j]
            if !isempty(y_js)
                {:μ_proposed => j} ~ Normal((sum(y_js)/σ²_j + κ * ξ)/(n_j/σ²_j + κ), sqrt(1/(n_j/σ²_j + κ)))
            end
        end
    elseif move == 2
        # update vars
        for j=1:k
            y_js = [gt_ys[i] for i=1:n if tr[:z => i] == j]
            n_j = length(y_js)
            μ_j = tr[:μ => j]
            if !isempty(y_js)
                {:σ²_proposed => j} ~ InverseGamma(α + n_j/2, β + sum((y_js .- μ_j).^2)/2)
            end
        end
    elseif move == 3
        # update allocations
        μs = [tr[:μ => j] for j=1:k]
        σ²s = [tr[:σ² => j] for j=1:k]
        for i=1:n
            y_i = gt_ys[i]
            p = [exp(logpdf(Normal(μ, sqrt(σ²)), y_i)) for (μ, σ²) in zip(μs, σ²s)] .* w
            {:z_proposed => i} ~ Categorical(p ./ sum(p))
        end
    elseif move == 4
        # update W
        counts = zeros(k)
        for i=1:n
            counts[tr[:z => i]] += 1
        end
        alpha = δ * ones(k) + counts
        # for j=1:k
        #     {:w_proposed => j} ~ Gamma(alpha[j], 1)
        # end
        α0 = sum(alpha)
        for j=1:k-1
            α0 -= alpha[j]
            {:phi_proposed => j} ~ Beta(alpha[j], α0)
        end
    elseif move == 5
        # merge
        r1 ~ DiscreteUniform(1,k)
        r2 ~ DiscreteUniform(1,k-1)
        j_star ~ DiscreteUniform(1,k-1)
    elseif move == 6
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
    else
        error("Unknown move.")
    end
    return move
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

function copy_everything(tt::TraceTransformation, old_trace::Dict{Any,Real}, new_trace::Dict{Any,Real})
    for addr in keys(old_trace)
        copy_at_address(tt, old_trace, new_trace, addr)
    end
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


function involution!(tt::TraceTransformation,
    old_model_trace::Dict{Any,Real}, old_proposal_trace::Dict{Any,Real},
    new_model_trace::Dict{Any,Real}, new_proposal_trace::Dict{Any,Real})

    move = read_discrete(tt, old_proposal_trace, :move)
    k = read_discrete(tt, old_model_trace, :k)
    # TODO get n
    n = length(gt_ys)

    if move == 1
        # update means
        copy_everything(tt, old_model_trace, new_model_trace) # overwrite means later        
        copy_at_address(tt, old_proposal_trace, new_proposal_trace, :move)
        for j=1:k
            if haskey(old_proposal_trace, (:μ_proposed => j))
                # do not have to be tracked / differentiated
                proposed_value = read_continuous(tt, old_proposal_trace, :μ_proposed => j)
                old_value = read_continuous(tt, old_model_trace, :μ => j)
                write_continuous(tt, new_model_trace, :μ => j, proposed_value)
                write_continuous(tt, new_proposal_trace, :μ_proposed => j, old_value)
            end
        end
    elseif move == 2
        # update vars
        copy_everything(tt, old_model_trace, new_model_trace) # overwrite vars later
        copy_at_address(tt, old_proposal_trace, new_proposal_trace, :move)
        for j=1:k
            if haskey(old_proposal_trace, (:σ²_proposed => j))
                # do not have to be tracked / differentiated
                proposed_value = read_continuous(tt, old_proposal_trace, :σ²_proposed => j)
                old_value = read_continuous(tt, old_model_trace, :σ² => j)
                write_continuous(tt, new_model_trace, :σ² => j, proposed_value)
                write_continuous(tt, new_proposal_trace, :σ²_proposed => j, old_value)
            end
        end
    elseif move == 3
        # update allocations
        copy_everything(tt, old_model_trace, new_model_trace) # overwrite allocations later
        copy_at_address(tt, old_proposal_trace, new_proposal_trace, :move)
        for j=1:n
            proposed_value = read_discrete(tt, old_proposal_trace, :z_proposed => j)
            old_value = read_discrete(tt, old_model_trace, :z => j)
            write_discrete(tt, new_model_trace, :z => j, proposed_value)
            write_discrete(tt, new_proposal_trace, :z_proposed => j, old_value)
        end
    elseif move == 4
        # update w
        copy_everything(tt, old_model_trace, new_model_trace) # overwrite w later
        copy_at_address(tt, old_proposal_trace, new_proposal_trace, :move)
        # for j=1:k
            # do not have to be tracked / differentiated
            # proposed_value = read_continuous(tt, old_proposal_trace, :w_proposed => j)
            # old_value = read_continuous(tt, old_model_trace, :w => j)
            # write_continuous(tt, new_model_trace, :w => j, proposed_value)
            # write_continuous(tt, new_proposal_trace, :w_proposed => j, old_value)
        # end
        for j=1:k-1
            proposed_value = read_continuous(tt, old_proposal_trace, :phi_proposed => j)
            old_value = read_continuous(tt, old_model_trace, :phi => j)
            write_continuous(tt, new_model_trace, :phi => j, proposed_value)
            write_continuous(tt, new_proposal_trace, :phi_proposed => j, old_value)
        end
    elseif move == 5
        # merge
        r1 = read_discrete(tt, old_proposal_trace, :r1) # 1..K
        r2 = read_discrete(tt, old_proposal_trace, :r2) # 1..K-1
        j_star = read_discrete(tt, old_proposal_trace, :j_star)
        j1, j2 = r1, r2 + (r2 >= r1)

        phi = [read_continuous(tt, old_model_trace, :phi => j) for j in 1:k-1]
        w_arr = get_w(phi)

        w1, w2 = w_arr[j1], w_arr[j2] # read_continuous(tt, old_model_trace, :w => j1), read_continuous(tt, old_model_trace, :w => j2) 
        μ1, μ2 = read_continuous(tt, old_model_trace, :μ => j1), read_continuous(tt, old_model_trace, :μ => j2) 
        σ1², σ2² = read_continuous(tt, old_model_trace, :σ² => j1), read_continuous(tt, old_model_trace, :σ² => j2) 
       
        w, μ, σ² = get_merge_params(w1, μ1, σ1², w2, μ2, σ2²)

        # perform merge
        write_discrete(tt, new_model_trace, :k, k-1)
        for i=1:n
            z = read_discrete(tt, old_model_trace, :z => i)
            if z == j1 || z == j2
                write_discrete(tt, new_model_trace, :z => i, j_star)
                write_discrete(tt, new_proposal_trace, :to_first => i, z == j1)
            else
                write_discrete(tt, new_model_trace, :z => i, merge_idx(z, j_star, j1, j2))
            end
        end

        w_arr_new = Vector{Real}(undef, k-1)
        w_arr_new[j_star] = w
        # write_continuous(tt, new_model_trace, :w => j_star, w)
        write_continuous(tt, new_model_trace, :μ => j_star, μ)
        write_continuous(tt, new_model_trace, :σ² => j_star, σ²)

        for j = 1:k
            if j != j1 && j != j2
                new_idx = merge_idx(j, j_star, j1, j2)
                w_arr_new[new_idx] = w_arr[j]
                # copy_at_addresses(tt, old_model_trace, :w => j, new_model_trace, :w => new_idx)
                copy_at_addresses(tt, old_model_trace, :μ => j, new_model_trace, :μ => new_idx)
                copy_at_addresses(tt, old_model_trace, :σ² => j, new_model_trace, :σ² => new_idx)
            end
        end
        phi_new = get_phi(w_arr_new)
        for j in eachindex(phi_new)
            write_continuous(tt, new_model_trace, :phi => j, phi_new[j])
        end

        # reverse split params
        u1, u2, u3 = reverse_split_params(w, μ, σ², w1, μ1, σ1², w2, μ2, σ2²)
        _w1, _w2, _μ1, _μ2, _σ1², _σ2² = get_split_params(w, μ, σ², u1, u2, u3)
        @assert _w1 ≈ w1
        @assert _w2 ≈ w2
        @assert _μ1 ≈ μ1
        @assert _μ2 ≈ μ2
        @assert _σ1² ≈ σ1²
        @assert _σ2² ≈ σ2²

        write_discrete(tt, new_proposal_trace, :move, 6)
        write_continuous(tt, new_proposal_trace, :u1, u1)
        write_continuous(tt, new_proposal_trace, :u2, u2)
        write_continuous(tt, new_proposal_trace, :u3, u3)
        write_discrete(tt, new_proposal_trace, :r1, r1)
        write_discrete(tt, new_proposal_trace, :r2, r2)
        write_discrete(tt, new_proposal_trace, :j_star, j_star)

        # reads 4 + k-1:
        # :μ => j1, :μ => j2, :σ² => j1, :σ² => j2, :phi => j j=1...(k-1), 
        # writes 5 + k-2
        # :u1, :u2, :u3, :μ => j_star, :μ => j_star, :σ² => j1, :phi => j j=1...(k-2)

    elseif move == 6
        # split
        r1 = read_discrete(tt, old_proposal_trace, :r1) # 1..K+1
        r2 = read_discrete(tt, old_proposal_trace, :r2) # 1..K
        j_star = read_discrete(tt, old_proposal_trace, :j_star)
        u1 = read_continuous(tt, old_proposal_trace, :u1)
        u2 = read_continuous(tt, old_proposal_trace, :u2)
        u3 = read_continuous(tt, old_proposal_trace, :u3)

        j1, j2 = r1, r2 + (r2 >= r1)


        phi = [read_continuous(tt, old_model_trace, :phi => j) for j in 1:k-1]
        w_arr = get_w(phi)

        w = w_arr[j_star] # read_continuous(tt, old_model_trace, :w => j_star)
        μ = read_continuous(tt, old_model_trace, :μ => j_star)
        σ² = read_continuous(tt, old_model_trace, :σ² => j_star)

        w1, w2, μ1, μ2, σ1², σ2² = get_split_params(w, μ, σ², u1, u2, u3)

        # perform split
        write_discrete(tt, new_model_trace, :k, k+1)
        for i=1:n
            z = read_discrete(tt, old_model_trace, :z => i)
            if z == j_star
                to_first = read_discrete(tt, old_proposal_trace, :to_first => i)
                write_discrete(tt, new_model_trace, :z => i, to_first ? j1 : j2)
            else
                write_discrete(tt, new_model_trace, :z => i, split_idx(z, j_star, j1 ,j2))
            end
        end

        w_arr_new = Vector{Real}(undef, k+1)
        w_arr_new[j1] = w1
        w_arr_new[j2] = w2

        # write_continuous(tt, new_model_trace, :w => j1, w1)
        write_continuous(tt, new_model_trace, :μ => j1, μ1)
        write_continuous(tt, new_model_trace, :σ² => j1, σ1²)

        # write_continuous(tt, new_model_trace, :w => j2, w2)
        write_continuous(tt, new_model_trace, :μ => j2, μ2)
        write_continuous(tt, new_model_trace, :σ² => j2, σ2²)

        for j=1:k
            if j != j_star
                new_idx = split_idx(j, j_star, j1, j2)
                w_arr_new[new_idx] = w_arr[j]
                # copy_at_addresses(tt, old_model_trace, :w => j, new_model_trace, :w => new_idx)
                copy_at_addresses(tt, old_model_trace, :μ => j, new_model_trace, :μ => new_idx)
                copy_at_addresses(tt, old_model_trace, :σ² => j, new_model_trace, :σ² => new_idx)
            end
        end

        phi_new = get_phi(w_arr_new)
        for j in eachindex(phi_new)
            write_continuous(tt, new_model_trace, :phi => j, phi_new[j])
        end


        # reverse merge params
        _w, _μ, _σ² = get_merge_params(w1, μ1, σ1², w2, μ2, σ2²)
        @assert _w ≈ w
        @assert _μ ≈ μ
        @assert _σ² ≈ σ²

        write_discrete(tt, new_proposal_trace, :move, 5)
        write_discrete(tt, new_proposal_trace, :r1, r1)
        write_discrete(tt, new_proposal_trace, :r2, r2)
        write_discrete(tt, new_proposal_trace, :j_star, j_star)

        # reads 5 + k-1
        # :u1, :u2, :u3, :μ => j_star, :μ => j_star, :σ² => j1, :phi => j j=1...(k-1)
        # writes 4 + k:
        # :μ => j1, :μ => j2, :σ² => j1, :σ² => j2, :phi => j j=1...k, 
    end
end

Random.seed!(0)
@time traces, lp = imcmc(
    gmm, (n,), observations,
    aux_model, (n,),
    involution!,
    5000 * 6;
    check_involution=true
);
sum(traces[:k,i] != traces[:k,i+1] for i in 1:(length(traces)-1))
maximum(lp)
best_trace = traces.data[argmax(lp)]


import PyPlot
function visualize_trace(tr; color_shift=0, raw=false, path="plot.pdf")
    gaussian_pdf(μ, σ², w) = x -> w * exp(logpdf(Normal(μ, sqrt(σ²)), x));

    n, k = length(gt_ys), tr[:k]
    phi = [tr[:phi=>j] for j in 1:(k-1)]
    w_arr = get_w(phi)

    cmap = PyPlot.get_cmap("Paired")
    p = PyPlot.figure()
    
    for j=1:k
        y_js = [gt_ys[i] for i=1:n if tr[:z=>i] == j]
        μ, σ² = tr[:μ=>j], tr[:σ²=>j]
        w = w_arr[j]
        PyPlot.hist(y_js, density=true, bins=6, color=cmap(2j-2 + 2color_shift), alpha=0.5)
        
        dom = (μ - 3sqrt(σ²)):1e-1:μ + 3sqrt(σ²)
        PyPlot.plot(dom, gaussian_pdf(μ, σ², w).(dom), color=cmap(2j-1+ 2color_shift))
        
        if j <= gt_k
            dom = (gt_μs[j] - 3sqrt(gt_σ²s[j])):1e-1:gt_μs[j] + 3sqrt(gt_σ²s[j])
            PyPlot.plot(dom, gaussian_pdf(gt_μs[j], gt_σ²s[j], gt_ws[j]).(dom), color="gray")
        end
        PyPlot.plot(y_js, 5e-3 .+ zeros(length(y_js)), ".", color=cmap(2j-1 + 2color_shift))
        PyPlot.xlabel("x"); PyPlot.ylabel("density");
    end
    if raw
        PyPlot.plot(gt_ys, 0.25 .+ zeros(n), "o", color="black", alpha=0.5)
    end
    # PyPlot.savefig(path)
end

for k in unique(traces[:k])
    mask = traces[:k] .== k
    amax = argmax(lp[mask])
    println("k=$k, lp=$(lp[mask][amax])")
    best_trace_k = traces.data[mask][amax]
    visualize_trace(best_trace_k)
    display(PyPlot.gcf())
end

function plot_lps(lps;path="plot.pdf")
    p = PyPlot.figure()
    PyPlot.plot(lps)
    PyPlot.ylabel("Log Probability")
    # PyPlot.savefig(path)
end

plot_lps(lp[10:end])
PyPlot.gcf()




transformation = TraceTransformation(involution!)

sampler = Evaluation.TraceSampler()
gmm((n,), sampler, observations)
old_model_trace = sampler.X
old_p = sampler.W

sampler = Evaluation.TraceSampler()
aux_model((old_model_trace,n), sampler, Dict())
old_proposal_trace = sampler.X
old_q = sampler.W

new_model_trace, new_proposal_trace = apply(transformation, old_model_trace, old_proposal_trace)
J = Evaluation.jacobian_fwd_diff(transformation, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace)
det(J)
old_model_trace_2, old_proposal_trace_2 = apply(transformation, new_model_trace, new_proposal_trace)
all(old_model_trace[addr] ≈ old_model_trace_2[addr] for addr in keys(old_model_trace)) && all(old_proposal_trace[addr] ≈ old_proposal_trace[addr] for addr in keys(old_proposal_trace))


import Distributions

d = Distributions.Dirichlet([2., 3.])
x = rand(d)
logpdf(d, x)

logpdf(Gamma(2., 1.), x[1]) + logpdf(Gamma(3., 1.), x[2])


c = [1, 2, 4, 8]
cat_obs = Dict{Any,Real}()
begin
    i = 0
    for (k, ck) in enumerate(c)
        for _ in 1:ck
            i += 1
            cat_obs[:x => i] = k
        end
    end
end

@ppl function cat(k, n)
    w = @subppl dirichlet(1., k)
    for i = 1:n
        {:x => i} ~ Categorical(w)
    end
end
posterior = Distributions.Dirichlet(1. .+ c)


Random.seed!(0)
@time result = likelihood_weighting(cat, (length(c), length(cat_obs)), cat_obs, 1_000_000);
traces = Evaluation.UniversalTraces(result[1])
W = exp.(result[3])

for i in 1:length(c)
    println(i, ": ", traces[:w => i]'W)
end

i = 3
post_mean = traces[:w => i]'W
gamma = Distributions.Gamma(1. + c[i], post_mean / (1. + c[i]))
mean(gamma)

g = rand(gamma, 1_000_000)
histogram(g, normalize=true, linewidth=0, alpha=0.5, xlims=(0.,5))
histogram!(traces[:w => i], weights=W, normalize=true, linewidth=0, alpha=0.5)


function rand_gamma(α, N)
    K = length(α)
    V = Array{Float64}(undef, K, N)
    for i in 1:K
        V[i,:] = rand(Distributions.Gamma(α[i], 1.), N)
    end
    return V ./ sum(V, dims=1)
end

function beta_to_dir!(phi, x)
    K = length(x)
    s = 0. # sum_{i=1}^{j-1} x[i]
    for j in 1:K-1
        x[j] = (1-s)*phi[j]
        s += x[j]
    end
    x[K] = 1. - s
    return x
end

function beta_to_dir_jacobian(phi, x)
    K = length(phi)
    J = zeros(K+1, K)
    # J[j,k] = ∂x[j] / ∂phi[k]
    # for j in 1:K
    #     J[j,j] = (1 - sum(x[i] for i in 1:(j-1); init=0.))
    # end
    for k in 1:K, j in (k+1):K
        J[j,k] = -sum(J[i,k] for i in 1:k; init=0.) * phi[j]
    end
    for k in 1:K
        J[K+1,k] = -sum(J[i,k] for i in 1:k; init=0.)
    end
    return J
end

function dir_to_beta!(phi, x)
    K = length(x)
    s = 0. # sum_{i=1}^{j-1} x[i]
    for j in 1:K-1
        phi[j] = x[j] / (1-s)
        s += x[j]
    end
    return phi
end



import Tracker, ForwardDiff
import LinearAlgebra: det, inv
begin
    α = 1. .+ c
    α0 = sum(α)
    K = length(α)
    phi = Array{Float64}(undef, K-1)
    x = Array{Float64}(undef, K)

    for j in 1:K-1
        α0 -= α[j]
        phi[j] = rand(Distributions.Beta(α[j], α0))
    end
    beta_to_dir!(phi, x)
    phi, x
end
dir_to_beta!(copy(phi), copy(x))

J = ForwardDiff.jacobian(phi -> beta_to_dir!(phi, Array{Real}(undef, K)), phi)

beta_to_dir_jacobian(phi, x)

function rand_beta(α, N)
    K = length(α)
    α0 = sum(α)
    phi = Array{Float64}(undef, K-1, N)
    x = Array{Float64}(undef, K, N)

    for j in 1:K-1
        α0 -= α[j] # sum_{i=j+1}^K α[i]
        phi[j,:] = rand(Distributions.Beta(α[j], α0), N)
    end
    for i in 1:N
        beta_to_dir!(view(phi,:,i), view(x,:,i))
    end
    return x
end

α = 1. .+ c
dirichlet = Distributions.Dirichlet(α)
D = rand(dirichlet, 1_000_000)
G = rand_gamma(α, 1_000_000)
B = rand_beta(α, 1_000_000)

begin
    i = 4
    histogram(D[i,:], normalize=true, linewidth=0, alpha=0.5, legend=false)
    histogram!(G[i,:], normalize=true, linewidth=0, alpha=0.5, legend=false)
    histogram!(B[i,:], normalize=true, linewidth=0, alpha=0.5, legend=false)
end

using Turing
using Plots

@model function cat_turing(k, x)
    w ~ Dirichlet(ones(k))
    for i in eachindex(x)
        x[i] ~ Categorical(w)
    end
end
x = collect(values(cat_obs))
m = cat_turing(length(c), x)
posterior = Dirichlet(1. .+ c)

Turing.Random.seed!(0)
result = sample(m, MH(:w => posterior), 1_000_000);

i = 1
histogram(result["w[$i]"], normalize=true, linewidth=0, alpha=0.5)

post = rand(posterior, 1_000_000)
histogram!(post[i,:], normalize=true, linewidth=0, alpha=0.5)

@model function cat_beta_turing(k, x)
    α = ones(k)
    α0 = sum(α)
    phi = Vector{Float64}(undef, k-1)
    for j in 1:k-1
        α0 -= α[j] # sum_{i=j+1}^K α[i]
        phi[j] ~ Beta(α[j], α0)
    end
    w = Vector{Float64}(undef, k)
    beta_to_dir!(phi, w)
    for i in eachindex(x)
        x[i] ~ Categorical(w)
    end
end


x = collect(values(cat_obs))
m = cat_beta_turing(length(c), x)
α_post = 1. .+ c
posterior = Dirichlet(α_post)

Turing.Random.seed!(0)
result = sample(m, IS(), 1_000_000);



post = rand(posterior, 1_000_000)
post_transformed = zeros(size(post,1)-1, size(post,2))
for i in 1:1_000_000
    dir_to_beta!(view(post_transformed,:,i), view(post,:,i))
end

begin
    post_beta = zeros(size(post,1)-1, size(post,2))
    α0 = sum(α_post)
    for j in 1:length(α_post) -1
        α0 -= α_post[j] # sum_{i=j+1}^K α[i]
        post_beta[j,:] = rand(Beta(α_post[j], α0), 1_000_000)
    end
    post_beta
end


i = 2
histogram(result["phi[$i]"], weights=exp.(result[:lp]), normalize=true, linewidth=0, alpha=0.5)
histogram!(post_transformed[i,:], normalize=true, linewidth=0, alpha=0.5)
histogram!(post_beta[i,:], normalize=true, linewidth=0, alpha=0.5)
