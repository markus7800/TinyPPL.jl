
include("../gmm/common.jl");

function to_plotting_format(tr)
    tr = copy(tr)

    n, k = length(gt_ys), tr[:k]
    phi = [tr[:phi=>j] for j in 1:(k-1)]
    w_arr = get_w(phi)

    for j in 1:k
        tr[:w => j] = w_arr[j]
    end

    return tr
end