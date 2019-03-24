# update_weights.jl

export lω_to_v, v_to_lω;

function lω_to_v(lω::Array{T, 1}) where T <: Real
    H = length(lω)
    H > 2 || throw(ArgumentError("H must be at least 3."))

    v = Array{T, 1}(undef, H-1)

    v[1] = exp(lω[1])
    cumsuml1minusv = log(1.0 - v[1])

    for h = 2:(H-1)
        v[h] = exp(lω[h] - cumsuml1minusv)
        cumsuml1minusv += log(1.0 - v[h])
    end
    cumsuml1minusv ≈ lω[H] || throw(error("lω_to_v unstable."))

    return v
end

function v_to_lω(v::Array{T, 1}) where T <: Real
    H = length(v) + 1
    H > 2 || throw(ArgumentError("H must be at least 3."))

    lω = Array{T, 1}(undef, H)

    lω[1] = log(v[1])
    cumsuml1minusv = log(1.0 - v[1])

    for h = 2:(H-1)
        lω[h] = log(v[h]) + cumsuml1minusv
        cumsuml1minusv += log(1.0 - v[h])
    end

    lω[H] = cumsuml1minusv

    return lω
end

function lfc_v(v::Array{T, 1}, a_v::Array{T, 1}, b_v::Array{T, 1},
    lNX_mat::Array{T, 2}) where T <: Real
    ### log full (block) conditional for v

    lω = v_to_lω(v)

    lωNX_vec = lωNXvec(lω, lNX_mat)

    ## replaced by lωXvec
    # lωNX_mat = broadcast(+, lω, lNX_mat') # H by n
    # lωNX_vec = BayesInference.logsumexp(lωNX_mat, 1) # n vector

    ldb = [ logpdf( Beta(a_v[h], b_v[h]), v[h] ) for h = 1:length(v) ]

    return sum(ldb) - sum(lωNX_vec), lωNX_vec
end

function mvSlice_v(v::Array{T, 1}, a_v::Array{T, 1}, b_v::Array{T, 1}, lNX_mat::Array{T, 2},
    w_slice::T=1.0, maxtries::Int=1000) where T <: Real

    ### Multivariate slice sampler Figure 8 from Neal (2003)

    HH = length(v)

    ### Step A: Define the slice

    lf0, trash = lfc_v(v, a_v, b_v, lNX_mat)
    ee = rand(Exponential(1.0))
    zz = lf0 - ee

    ### Step B: Randomly position hyperrectangle

    U = rand(HH)
    L = max.(v .- w_slice .* U, 0.0 + eps(T))
    R = min.(L .+ w_slice, 1.0 - eps(T))

    ### Step C: Sample from hyperrectangle, shrinking when rejected

    keeptrying = true
    tries = 1
    cand = Array{T, 1}(undef, HH) # define outside scope of while loop
    lωNX_vec_out = Array{T, 1}(undef, length(trash))

    while keeptrying

        cand = L .+ rand(HH) .* (R .- L)
        lfcand, lωNX_vec_out = lfc_v(cand, a_v, b_v, lNX_mat)

        keeptrying = zz > lfcand

        if keeptrying

            tries += 1
            tries <= maxtries || throw(error("Exceeded maximum slice attempts."))

            for h = 1:HH
                if cand[h] < v[h]
                    L[h] = copy(cand[h])
                else
                    R[h] = copy(cand[h])
                end
                ## equivalent to
                # cand[h] < v[h] ? L[h] = copy(cand[h]) : R[h] = copy(cand[h])
            end
        end

    end

    return cand, lωNX_vec_out
end

function update_vlω_mvSlice!(model::Model_DPmRegJoint) where T <: Real

    M = StatsBase.counts(model.state.S, 1:model.H)
    a_v = 1.0 .+ M[1:(model.H-1)]
    b_v = reverse( model.state.α .+ cumsum( reverse( M[2:model.H] ) ) )

    if model.state.γδc == Inf && sum(model.state.γ) == 0
        v_new = [ rand( Beta(a_v[h], b_v[h]) ) for h = 1:(model.H-1) ]
    else
        v_new, lωNX_vec_new = mvSlice_v(model.state.v, a_v, b_v, model.state.lNX)
        model.state.lωNX_vec = lωNX_vec_new
    end

    lω_new = v_to_lω(v_new)

    model.state.v = v_new
    model.state.lω = lω_new

    return nothing
end
