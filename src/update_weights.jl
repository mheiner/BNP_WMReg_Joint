# update_weights.jl

export ω_to_v, v_to_ω;

function ω_to_v(ω::Array{T, 1}) where T <: Real
    H = length(ω)
    H > 2 || throw(ArgumentError("H must be at least 3."))

    v = Array{T, 1}(undef, H-1)

    v[1] = copy(ω[1])
    whatsleft = 1.0 - v[1]

    for h = 2:(H-1)
        v[h] = ω[h] / whatsleft
        whatsleft -= ω[h]
    end

    return v
end

# ω2v(log.([0.2, 0.3, 0.25, 0.15, 0.10]))

function v_to_ω(v::Array{T, 1}) where T <: Real
    H = length(v) + 1
    H > 2 || throw(ArgumentError("H must be at least 3."))

    ω = Array{T, 1}(undef, H)

    ω[1] = copy(v[1])
    whatsleft = 1.0

    for h = 2:(H-1)
        ω[h] = v[h] * whatsleft
        whatsleft -= ω[h]
    end

    ω[H] = whatsleft

    return ω
end

function mvSlice_v() where T <: Real

end

function update_vω_mvSlice!(model::Model_DPmRegJoint, lNX_mat::Array{T, 2}) where T <: Real

    M = StatsBase.counts(model.state.S, 1:model.H)
    a_v = 1.0 .+ M[1:(model.H-1)]
    b_v = reverse( model.state.α .+ cumsum( reverse( M[2:model.H] ) ) )

    v_new = mvSlice_v(model.state.v, a_v, b_v, lNX_mat, 1.0, 1000)
    ω_new = v2

    return nothing
end
