# compute full joint distribution

# X_t = v_1 + a v_2 + a^2 v_3 ..., v_i ~ Normal(0,σ_v)
# cov(X + Y, W + V) = cov(X,W) + cov(X,V) + cov(Y,W) + cov(Y,V)
# cov(X_t, X_{t+1}) = cov(X_t,a X_t + v_{t+1}) = a cov(X_t, X_t)



# [x_1, y_1, x_2, y_2, ..., x_T, y_T]
function get_joint_normal(T)
    Σ = zeros(2*T, 2*T)
    for t in 1:T
        x_ix = 2*(t-1) + 1
        y_ix = x_ix + 1
        Σ[x_ix, x_ix] = σ_v^2 * ((a^2)^t - 1) / (a^2 - 1) 
        Σ[y_ix, y_ix] = Σ[x_ix, x_ix] + σ_e^2
        Σ[x_ix, y_ix] = Σ[x_ix, x_ix]

        for (n,t2) in enumerate(t+1:T)
            x_ix_2 = 2*(t2-1) + 1
            y_ix_2 = x_ix_2 + 1
            Σ[x_ix, x_ix_2] = a^n * Σ[x_ix, x_ix]

            Σ[x_ix, y_ix_2] = Σ[x_ix, x_ix_2]

            Σ[y_ix, x_ix_2] = Σ[x_ix, x_ix_2]

            Σ[y_ix, y_ix_2] = Σ[x_ix, x_ix_2]
        end
    end
    for i in 1:2*T, j in (i+1):2*T
        Σ[j,i] = Σ[i,j]
    end

    reorder = vcat([2*(t-1) + 1 for t in 1:T], [2*(t-1) + 1 for t in 1:T] .+ 1)
    # [x_1, x_2, ..., x_T, y_1, y_2, ..., y_T]
    mu = zeros(2*T)
    Σ  = Σ[reorder,reorder]
    return mu, Σ
end

function get_true_posterior(T, y)
    mu, Σ = get_joint_normal(T)
    mu1 = mu[1:T]
    mu2 = mu[T+1:end]

    Σ11 = Σ[1:T,1:T]
    Σ12 = Σ[1:T,T+1:end]
    Σ21 = Σ[T+1:end,1:T]
    Σ22 = Σ[T+1:end,T+1:end]

    Σ_post = Σ11 - (Σ12 * (Σ22 \ Σ21))
    mu_post = (mu1 + Σ12 * (Σ22 \ (y - mu2)))

    return mu_post, Σ_post
end