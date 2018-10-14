# update_alloc.jl

export llik_numerator;

function llik_numerator(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Array{T, 1},
    lω::Array{T, 1}) where T <: Real

    yX = hcat(y, X)

    return llik_numerator(yX, K, H, μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, lω)
end
function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Array{T, 1},
    lω::Array{T, 1}) where T <: Real

    βγ_y = β_y_modify_γ(β_y, γ)

    if K > 1
        βγ_x, δγ_x = βδ_x_modify_γ(β_x, δ_x, γ, γδc)
    else
        δγ_x = δ_x_modify_γ(δ_x, γ, γδc)
    end

    μ = hcat(μ_y, μ_x)
    βγ = ( K > 1 ? [βγ_y, βγ_x...] : [βγ_y] )
    δγ = hcat(δ_y, δγ_x)

    # the rest could be done in parallel
    lW = hcat([ lω[h] .+
                lNX_sqfChol( Matrix(yX'), μ[h,:], [ βγ[k][h,:] for k = 1:K ], δγ[h,:] )
                for h = 1:H
              ]...) # lW is a n by H matrix

    return lW
end

function update_alloc!(model::Model_DPmRegJoint, yX::Array{T,2}) where T <: Real

    lW = llik_numerator(yX, model.K, model.H,
            model.state.μ_y, model.state.β_y, model.state.δ_y,
            model.state.μ_x, model.state.β_x, model.state.δ_x,
            model.state.γ, model.state.γδc, model.state.lω)

    ms = maximum(lW, dims=2) # maximum across columns
    bc_lWmimusms = broadcast(-, lW, ms)
    W = exp.(bc_lWmimusms)

    alloc_new = [ sample(StatsBase.Weights(W[i,:])) for i = 1:model.n ]
    model.state.S = alloc_new

    model.state.n_occup = length(unique(alloc_new))

    return lW # for llik calculation
end
