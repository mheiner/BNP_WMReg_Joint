# update_alloc.jl

export llik_numerator, llik_numerator_Σx_diag;

function llik_numerator(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Union{Float64, Array{T, 1}, Nothing},
    lω::Array{T, 1}) where T <: Real

    if isnothing(β_x) && K > 1 # indicates that Σx_type == :diag
        return llik_numerator_Σx_diag(y, X, K, H, μ_y, β_y, δ_y, μ_x, δ_x, γ, lω)
    else
        yX = hcat(y, X)
        return llik_numerator(yX, K, H, μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, lω)
    end
end
function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Array{T, 1}, # inflated variance method
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
function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Float64, # γδc == Inf (subset method)
    lω::Array{T, 1}) where T <: Real

    γδc == Inf || throw("Subset method for variable selection requires γδc = Inf.")
    size(yX,2) == (K+1) || throw("llik_numerator assumes a full X matrix.")

    γindx = findall(γ)
    nγ = length(γindx)

    if nγ == 0

        lW = hcat( [ lω[h] .+ logpdf.( Normal(μ_y[h], sqrt(δ_y[h])), yX[:,1] )  for h = 1:H ]... )

    elseif nγ > 0

        βγ_y = deepcopy(β_y[:, γindx]) # H by nγ matrix

        if nγ > 1
            βγ_x, δγ_x = βδ_x_modify_γ(β_x, δ_x, γ, γδc)
        else
            δγ_x = δ_x_modify_γ(δ_x, γ, γδc)
        end

        μγ = hcat(μ_y, μ_x[:,γindx])
        βγ = ( nγ > 1 ? [βγ_y, βγ_x...] : [βγ_y] )
        δγ = hcat(δ_y, δγ_x)

        yXγ = yX[:,vcat(1, (γindx .+ 1))]

        # the rest could be done in parallel
        lW = hcat([ lω[h] .+
            lNX_sqfChol( Matrix(yXγ'), μγ[h,:], [ βγ[k][h,:] for k = 1:nγ ], δγ[h,:] )
                for h = 1:H ]...) # lW is a n by H matrix
        end

    return lW # lW is a n by H matrix
end
function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Nothing, # γδc == nothing (integration method)
    lω::Array{T, 1}) where T <: Real

    size(yX,2) == (K+1) || throw("llik_numerator assumes a full X matrix.")

    γindx = findall(γ)
    γindx_withy = vcat(1, γindx .+ 1)
    nγ = length(γindx)

    if nγ == 0

        lW = hcat( [ lω[h] .+ logpdf.( Normal(μ_y[h], sqrt(δ_y[h])), yX[:,1] )  for h = 1:H ]... )

    elseif nγ > 0

        βγ_y = β_y_modify_γ(β_y, γ) # H by K matrix

        μ = hcat(μ_y, μ_x)
        β = ( K > 1 ? [βγ_y, β_x...] : [βγ_y] )
        δ = hcat(δ_y, δ_x)

        # the rest could be done in parallel
        Σxs = [ PDMat( sqfChol_to_Σ( [ β[k][h,:] for k = 1:K ], δ[h,:] ).mat[γindx_withy, γindx_withy] ) for h = 1:H ]
        lW = hcat( [ lω[h] .+ logpdf(MultivariateNormal(μ[h,γindx_withy], Σxs[h]), Matrix(yX[:,γindx_withy]')) for h = 1:H ]... ) # n by H matrix

        end

    return lW # lW is a n by H matrix
end

function llik_numerator_Σx_diag(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, # compatible with both subset and integration methods
    lω::Array{T, 1}) where T <: Real

    n = length(y)
    γindx = findall(γ)

    lW = zeros(Float64, n, H)

    for i = 1:n
        for h = 1:H
            mean_y = deepcopy(μ_y[h])
            for k in γindx
                mean_y -= β_y[h, k] * (X[i,k] - μ_x[h, k])
                lW[i,h] += -0.5*log( 2.0π * δ_x[h,k] ) - 0.5*( X[i,k] - μ_x[h,k] )^2 / δ_x[h,k]
            end
            lW[i,h] += -0.5*log( 2.0π * δ_y[h] ) - 0.5*( y[i] - mean_y )^2 / δ_y[h]
            lW[i,h] += lω[h]
        end
    end

    return lW # lW is a n by H matrix
end

function update_alloc!(model::Model_BNP_WMReg_Joint, yX::Array{T,2}) where T <: Real

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
function update_alloc!(model::Model_BNP_WMReg_Joint, y::Array{T,1}, X::Array{T,2}) where T <: Real

    lW = llik_numerator(y, X, model.K, model.H,
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
