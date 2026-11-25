# ===== involution model ===========================================================================

function copy_everything(tt::TraceTransformation, old_trace::UniversalTrace, new_trace::UniversalTrace)
    for addr in keys(old_trace)
        copy_at_address(tt, old_trace, new_trace, addr)
    end
end

function mh_means_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace,
    k::Int, n::Int)
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
end

function mh_vars_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace,
    k::Int, n::Int)
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
end

function mh_allocations_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace,
    k::Int, n::Int)
    # update allocations
    copy_everything(tt, old_model_trace, new_model_trace) # overwrite allocations later
    copy_at_address(tt, old_proposal_trace, new_proposal_trace, :move)
    for j=1:n
        proposed_value = read_discrete(tt, old_proposal_trace, :z_proposed => j)
        old_value = read_discrete(tt, old_model_trace, :z => j)
        write_discrete(tt, new_model_trace, :z => j, proposed_value)
        write_discrete(tt, new_proposal_trace, :z_proposed => j, old_value)
    end
end

function mh_w_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace,
    k::Int, n::Int)
    # update w
    copy_everything(tt, old_model_trace, new_model_trace) # overwrite w later
    copy_at_address(tt, old_proposal_trace, new_proposal_trace, :move)
    for j=1:k-1
        proposed_value = read_continuous(tt, old_proposal_trace, :phi_proposed => j)
        old_value = read_continuous(tt, old_model_trace, :phi => j)
        write_continuous(tt, new_model_trace, :phi => j, proposed_value)
        write_continuous(tt, new_proposal_trace, :phi_proposed => j, old_value)
    end
end

function merge_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace,
    k::Int, n::Int)
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

    write_discrete(tt, new_proposal_trace, :move, 5)
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
end

function split_involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace,
    k::Int, n::Int)
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

    write_discrete(tt, new_proposal_trace, :move, 6)
    write_discrete(tt, new_proposal_trace, :r1, r1)
    write_discrete(tt, new_proposal_trace, :r2, r2)
    write_discrete(tt, new_proposal_trace, :j_star, j_star)

    # reads 5 + k-1
    # :u1, :u2, :u3, :μ => j_star, :μ => j_star, :σ² => j1, :phi => j j=1...(k-1)
    # writes 4 + k:
    # :μ => j1, :μ => j2, :σ² => j1, :σ² => j2, :phi => j j=1...k, 
end

function involution!(tt::TraceTransformation,
    old_model_trace::UniversalTrace, old_proposal_trace::UniversalTrace,
    new_model_trace::UniversalTrace, new_proposal_trace::UniversalTrace)

    move = read_discrete(tt, old_proposal_trace, :move)
    k = read_discrete(tt, old_model_trace, :k)
    n = length(gt_ys) # TODO get n

    if move == 1
        mh_means_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, k, n)
    elseif move == 2
        mh_vars_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, k, n)
    elseif move == 3
        mh_allocations_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, k, n)
    elseif move == 4
        mh_w_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, k, n)
    elseif move == 5
        split_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, k, n)
    elseif move == 6
        merge_involution!(tt, old_model_trace, old_proposal_trace, new_model_trace, new_proposal_trace, k, n)
    end
end