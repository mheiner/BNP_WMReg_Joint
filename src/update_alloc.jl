# update_alloc.jl

export llik_numerator, llik_numerator_Σx_diag;

function lNymat(model::Model_BNP_WMReg_Joint)

    out = zeros(Float64, model.n, model.H)

    if model.γ_type in (:fixed, :global)
        γ_indx = findall(model.state.γ)
    end

    for h = 1:model.H

        if model.γ_type == :local
            γ_indx = findall(model.state.γ[h,:])
        end

        σ2 = deepcopy(model.state.δ_y[h])
        aa = -0.5 * log( 2.0π * σ2 )

        for i = 1:model.n

            μ = deepcopy(model.state.μ_y[h])

            for k in γ_indx
                μ -= model.state.β_y[h, k] * (model.X[i,k] - model.state.μ_x[h, k])
            end

            out[i,h] = aa - 0.5*( model.y[i] - μ )^2 / σ2

        end
    end

    return out
end

function llik_numerator(model::Model_BNP_WMReg_Joint)

    lNy = lNymat(model)
    size(lNy) == size(model.state.lNX) == (model.n, model.H) || throw("lNy and lNX dimension mismatch. lNy: $(size(lNy)) and lNX: $(size(model.state.lNX))")

    ldens_num = lNy .+ model.state.lNX
    lW = broadcast(+, permutedims(model.state.lω), ldens_num)

    return lW # n by H matrix
end

## The alternates below are deprecated, but may be useful sometime?
# function llik_numerator(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
#     μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
#     μ_x::Array{T, 2},
#     β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
#     δ_x::Array{Float64,2},
#     γ::Union{BitArray{1}, BitArray{2}}, γδc::Union{Float64, Array{T, 1}, Nothing},
#     lω::Array{T, 1}) where T <: Real

#     if isnothing(β_x) && K > 1 # indicates that Σx_type == :diag
#         return llik_numerator_Σx_diag(y, X, K, H, μ_y, β_y, δ_y, μ_x, δ_x, γ, lω)
#     else
#         yX = hcat(y, X)
#         return llik_numerator(yX, K, H, μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, lω)
#     end
# end
# function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
#     μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
#     μ_x::Array{T, 2},
#     β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
#     δ_x::Array{Float64,2},
#     γ::BitArray{1}, γδc::Array{T, 1}, # inflated variance method
#     lω::Array{T, 1}) where T <: Real

#     βγ_y = β_y_modify_γ(β_y, γ)

#     if K > 1
#         βγ_x, δγ_x = βδ_x_modify_γ(β_x, δ_x, γ, γδc)
#     else
#         δγ_x = δ_x_modify_γ(δ_x, γ, γδc)
#     end

#     μ = hcat(μ_y, μ_x)
#     βγ = ( K > 1 ? [βγ_y, βγ_x...] : [βγ_y] )
#     δγ = hcat(δ_y, δγ_x)

#     # the rest could be done in parallel
#     lW = hcat([ lω[h] .+
#                 lNX_sqfChol( Matrix(yX'), μ[h,:], [ βγ[k][h,:] for k = 1:K ], δγ[h,:] )
#                 for h = 1:H
#               ]...) # lW is a n by H matrix

#     return lW
# end
# function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
#     μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
#     μ_x::Array{T, 2},
#     β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
#     δ_x::Array{Float64,2},
#     γ::BitArray{1}, γδc::Float64, # γδc == Inf (subset method)
#     lω::Array{T, 1}) where T <: Real

#     γδc == Inf || throw("Subset method for variable selection requires γδc = Inf.")
#     size(yX,2) == (K+1) || throw("llik_numerator assumes a full X matrix.")

#     γindx = findall(γ)
#     nγ = length(γindx)

#     if nγ == 0

#         lW = hcat( [ lω[h] .+ logpdf.( Normal(μ_y[h], sqrt(δ_y[h])), yX[:,1] )  for h = 1:H ]... )

#     elseif nγ > 0

#         βγ_y = deepcopy(β_y[:, γindx]) # H by nγ matrix

#         if nγ > 1
#             βγ_x, δγ_x = βδ_x_modify_γ(β_x, δ_x, γ, γδc)
#         else
#             δγ_x = δ_x_modify_γ(δ_x, γ, γδc)
#         end

#         μγ = hcat(μ_y, μ_x[:,γindx])
#         βγ = ( nγ > 1 ? [βγ_y, βγ_x...] : [βγ_y] )
#         δγ = hcat(δ_y, δγ_x)

#         yXγ = yX[:,vcat(1, (γindx .+ 1))]

#         # the rest could be done in parallel
#         lW = hcat([ lω[h] .+
#             lNX_sqfChol( Matrix(yXγ'), μγ[h,:], [ βγ[k][h,:] for k = 1:nγ ], δγ[h,:] )
#                 for h = 1:H ]...) # lW is a n by H matrix
#         end

#     return lW # lW is a n by H matrix
# end
# function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
#     μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
#     μ_x::Array{T, 2},
#     β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
#     δ_x::Array{Float64,2},
#     γ::BitArray{1}, γδc::Nothing, # γδc == nothing (integration method)
#     lω::Array{T, 1}) where T <: Real

#     size(yX,2) == (K+1) || throw("llik_numerator assumes a full X matrix.")

#     γindx = findall(γ)
#     γindx_withy = vcat(1, γindx .+ 1)
#     nγ = length(γindx)

#     if nγ == 0

#         lW = hcat( [ lω[h] .+ logpdf.( Normal(μ_y[h], sqrt(δ_y[h])), yX[:,1] )  for h = 1:H ]... )

#     elseif nγ > 0

#         βγ_y = β_y_modify_γ(β_y, γ) # H by K matrix

#         μ = hcat(μ_y, μ_x)
#         β = ( K > 1 ? [βγ_y, β_x...] : [βγ_y] )
#         δ = hcat(δ_y, δ_x)

#         # the rest could be done in parallel
#         Σxs = [ PDMat( sqfChol_to_Σ( [ β[k][h,:] for k = 1:K ], δ[h,:] ).mat[γindx_withy, γindx_withy] ) for h = 1:H ]
#         lW = hcat( [ lω[h] .+ logpdf(MultivariateNormal(μ[h,γindx_withy], Σxs[h]), Matrix(yX[:,γindx_withy]')) for h = 1:H ]... ) # n by H matrix

#         end

#     return lW # lW is a n by H matrix
# end

## local variable selection
function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{2}, # H by K bit matrix
    γδc::Float64, # γδc == Inf (subset method)
    lω::Array{T, 1}) where T <: Real

    γδc == Inf || throw("Subset method for variable selection requires γδc = Inf.")
    size(yX,2) == (K+1) || throw("llik_numerator assumes a full X matrix.")
    size(γ) == (H,K) || throw("Incorrect dimensions for γ.")

    lW = zeros(Float64, size(yX,1), H)
    lenβx = length(β_x)

    # this could be done in parallel
    for h = 1:H

        γindx = findall(γ[h,:])
        nγ = length(γindx)

        if nγ == 0

            lW[:,h] = lω[h] .+ logpdf.( Normal(μ_y[h], sqrt(δ_y[h])), yX[:,1] )

        elseif nγ > 0

            βγ_y = deepcopy(β_y[h, γindx]) # nγ length vector

            if nγ > 1
                βγ_x, δγ_x = βδ_x_h_modify_γ( [ β_x[j][h,:] for j = 1:lenβx ], δ_x[h,:], γ[h,:], γδc )
            else
                δγ_x = δ_x_h_modify_γ(δ_x[h,:], γ[h,:], γδc)
            end

            μγ = vcat(μ_y[h], μ_x[h,γindx])
            βγ = ( nγ > 1 ? [βγ_y, βγ_x...] : [βγ_y] )
            δγ = vcat(δ_y[h], δγ_x)

            yXγ = yX[:,vcat(1, (γindx .+ 1))]

            lW[:,h] =  lω[h] .+ lNX_sqfChol( Matrix(yXγ'), μγ, βγ, δγ )
        end

    end

    return lW # lW is a n by H matrix
end
function llik_numerator(yX::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{2}, # H by K bit matrix
    γδc::Nothing, # γδc == nothing (integration method)
    lω::Array{T, 1}) where T <: Real

    size(yX,2) == (K+1) || throw("llik_numerator assumes a full X matrix.")
    size(γ) == (H,K) || throw("Incorrect dimensions for γ.")

    lW = zeros(Float64, size(yX,1), H)
    lenβx = length(β_x)

    # this could be done in parallel
    for h = 1:H

        γindx = findall(γ[h,:])
        γindx_withy = vcat(1, γindx .+ 1)
        nγ = length(γindx)
    
        if nγ == 0

            lW[:,h] = lω[h] .+ logpdf.( Normal(μ_y[h], sqrt(δ_y[h])), yX[:,1] )

        elseif nγ > 0

            βγ_y = β_y_modify_γ(β_y[h,:], γ[h,:]) # length K vector

            μ = vcat(μ_y[h], μ_x[h,:])
            β = ( K > 1 ? [βγ_y, [ β_x[j][h,:] for j = 1:lenβx ]...] : [βγ_y] )
            δ = vcat(δ_y[h], δ_x[h,:])

            Σx = PDMat( sqfChol_to_Σ( β, δ ).mat[γindx_withy, γindx_withy] )
            lW[:,h] = lω[h] .+ logpdf(MultivariateNormal(μ, Σx), Matrix(yX[:,γindx_withy]'))
        end
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
function llik_numerator_Σx_diag(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    δ_x::Array{Float64,2},
    γ::BitArray{2}, # H by K bit matrix, compatible with both subset and integration methods
    lω::Array{T, 1}) where T <: Real

    size(γ) == (H,K) || throw("Incorrect dimensions for γ.")

    n = length(y)
    γindxes = [ findall(γ[h,:]) for h = 1:H ]

    lW = zeros(Float64, n, H)

    for i = 1:n
        for h = 1:H
            mean_y = deepcopy(μ_y[h])
            for k in γindxes[h]
                mean_y -= β_y[h, k] * (X[i,k] - μ_x[h, k])
                lW[i,h] += -0.5*log( 2.0π * δ_x[h,k] ) - 0.5*( X[i,k] - μ_x[h,k] )^2 / δ_x[h,k]
            end
            lW[i,h] += -0.5*log( 2.0π * δ_y[h] ) - 0.5*( y[i] - mean_y )^2 / δ_y[h]
            lW[i,h] += lω[h]
        end
    end

    return lW # lW is a n by H matrix
end

## Standard Gibbs
# function update_alloc!(model::Model_BNP_WMReg_Joint) where T <: Real

#     lW = llik_numerator(model)

#     ms = maximum(lW, dims=2) # maximum across columns
#     bc_lWmimusms = broadcast(-, lW, ms)
#     W = exp.(bc_lWmimusms)

#     alloc_new = [ sample(StatsBase.Weights(W[i,:])) for i = 1:model.n ]
#     model.state.S = deepcopy(alloc_new)

#     model.state.n_occup = length(unique(alloc_new))

#     return lW # for llik calculation
# end

## Metropolized full conditional update
# function update_alloc!(model::Model_BNP_WMReg_Joint) where T <: Real

#     lW = llik_numerator(model)

#     ms = maximum(lW, dims=2) # maximum across columns
#     bc_lWmimusms = broadcast(-, lW, ms)
#     W = exp.(bc_lWmimusms)

#     ## Metropolized discrete Gibbs
#     for i in 1:model.n

#         Si = deepcopy(model.state.S[i])
#         w_cand = deepcopy(W[i,:])
#         w_cand[Si] = 0.0
#         cand = sample(StatsBase.Weights(w_cand))
#         lar = logsumexp( bc_lWmimusms[i, 1:end .!= Si ] ) - logsumexp( bc_lWmimusms[i, 1:end .!= cand ] )

#         if log(rand()) < lar
#             model.state.S[i] = deepcopy(cand)
#         end

#     end

#     model.state.n_occup = length(unique(model.state.S))

#     return lW # for llik calculation
# end


## These alternates are deprecated, but may be useful sometime?
# function update_alloc!(model::Model_BNP_WMReg_Joint, yX::Array{T,2}) where T <: Real

#     lW = llik_numerator(yX, model.K, model.H,
#             model.state.μ_y, model.state.β_y, model.state.δ_y,
#             model.state.μ_x, model.state.β_x, model.state.δ_x,
#             model.state.γ, model.state.γδc, model.state.lω)

#     ms = maximum(lW, dims=2) # maximum across columns
#     bc_lWmimusms = broadcast(-, lW, ms)
#     W = exp.(bc_lWmimusms)

#     alloc_new = [ sample(StatsBase.Weights(W[i,:])) for i = 1:model.n ]
#     model.state.S = deepcopy(alloc_new)

#     model.state.n_occup = length(unique(alloc_new))

#     return lW # for llik calculation
# end
# function update_alloc!(model::Model_BNP_WMReg_Joint, y::Array{T,1}, X::Array{T,2}) where T <: Real

#     lW = llik_numerator(y, X, model.K, model.H,
#             model.state.μ_y, model.state.β_y, model.state.δ_y,
#             model.state.μ_x, model.state.β_x, model.state.δ_x,
#             model.state.γ, model.state.γδc, model.state.lω)

#     ms = maximum(lW, dims=2) # maximum across columns
#     bc_lWmimusms = broadcast(-, lW, ms)
#     W = exp.(bc_lWmimusms)

#     alloc_new = [ sample(StatsBase.Weights(W[i,:])) for i = 1:model.n ]
#     model.state.S = deepcopy(alloc_new)

#     model.state.n_occup = length(unique(alloc_new))

#     return lW # for llik calculation
# end


function lmargy(Λ::PDMat, a::T, b::T) where T <: Real
    return -0.5 * logdet(Λ) + lgamma(a) - a * log(b)
end

## Integrate out eta_y and Metropolize
function update_alloc!(model::Model_BNP_WMReg_Joint) where T <: Real

    ## These remain fixed throughout
    Λβ0star_ηy = model.state.Λ0star_ηy * model.state.β0star_ηy
    βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)

    if model.γ_type == :local
        D_vec = [ construct_Dh(h, model.X, model.state.μ_x[h,:], model.state.γ[h,:] ) for h = 1:model.H ] # this is evaluated for ALL i and h, so it never changes
    elseif model.γ_type in (:fixed, :global)
        D_vec = [ construct_Dh(h, model.X, model.state.μ_x[h,:], model.state.γ ) for h = 1:model.H ] # this is evaluated for ALL i and h, so it never changes
    end
    
    ## These will evolve
    indx_h_vec = [ findall(model.state.S .== h) for h = 1:model.H ]
    n_h_vec = [ length(indx_h_vec[h]) for h = 1:model.H ]

    ## Calculate Lam1, a1, b1 for all h; these will evolve
    a1_vec = 0.5 .* ( model.state.ν_δy .+ n_h_vec )
    Λ1star_ηy_vec = [ n_h_vec[h] > 0 ? get_Λ1star_ηy_h(D_vec[h][indx_h_vec[h],:], model.state.Λ0star_ηy) : deepcopy(model.state.Λ0star_ηy) for h = 1:model.H ]
    β1star00_vec = [ n_h_vec[h] > 0 ? (Λβ0star_ηy + D_vec[h][indx_h_vec[h],:]'model.y[indx_h_vec[h]]) : deepcopy(Λβ0star_ηy) for h = 1:model.H ]
    β1star_ηy_vec = [ n_h_vec[h] > 0 ? Λ1star_ηy_vec[h] \ β1star00_vec[h] : deepcopy(model.state.β0star_ηy) for h = 1:model.H ]
    b1_vec = [ n_h_vec[h] > 0 ? get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, model.y[indx_h_vec[h]], βΛβ0star_ηy, Λ1star_ηy_vec[h], β1star_ηy_vec[h]) : 0.5*(model.state.ν_δy * model.state.s0_δy)  for h = 1:model.H ]

    lmargy_n0 = lmargy(model.state.Λ0star_ηy, 0.5*model.state.ν_δy, 0.5*(model.state.ν_δy * model.state.s0_δy)) # this remains constant throughout
    lmargy_out = [ n_h_vec[h] > 0 ? lmargy(Λ1star_ηy_vec[h], a1_vec[h], b1_vec[h]) : deepcopy(lmargy_n0) for h = 1:model.H ] # this currently has y_i in its component

    for i = 1:model.n

        Si = model.state.S[i]
        
        ### collect Lam1, a1, b1 for component Si without y_i
        lmargy_in = deepcopy(lmargy_out)

        ## create tmps (these will reflect the 'alternate' state for each h=1:H)
        Λ_tmp_vec = deepcopy(Λ1star_ηy_vec)
        β1star00_tmp_vec = deepcopy(β1star00_vec)
        β1star_tmp_vec = deepcopy(β1star_ηy_vec)
        b1_tmp_vec = deepcopy(b1_vec)

        ## remove y_i from its component
        if n_h_vec[Si] == 1 # if it was the only one in the component
            lmargy_out[ Si ] = deepcopy(lmargy_n0)
            Λ_tmp_vec[ Si ] = deepcopy(model.state.Λ0star_ηy)
            β1star00_tmp_vec[ Si ] = deepcopy(Λβ0star_ηy)
            β1star_tmp_vec[ Si ] = deepcopy(model.state.β0star_ηy)
            b1_tmp_vec[ Si ] = 0.5*(model.state.ν_δy * model.state.s0_δy)
        else
            d_tmp = deepcopy(D_vec[Si][i,:])
            Λ_tmp_vec[ Si ] = PDMat( Λ_tmp_vec[ Si ] + (-1.0) .* d_tmp * d_tmp' ) # PDMats doesn't have a subtract function
            β1star00_tmp_vec[ Si ] -= (d_tmp .* model.y[i])
            β1star_tmp_vec[ Si ] = Λ_tmp_vec[ Si ] \ β1star00_tmp_vec[ Si ]
            b1_tmp_vec[ Si ] = get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, model.y[setdiff(indx_h_vec[Si], i)], βΛβ0star_ηy, 
                Λ_tmp_vec[ Si ], β1star_tmp_vec[ Si ] )
            lmargy_out[ Si ] = lmargy(Λ_tmp_vec[ Si ], a1_vec[ Si ] - 0.5,  b1_tmp_vec[ Si ])
        end

        # now lmargy_out and all tmps are as though y_i didn't exist

        ### evaluate all other Lam1, a1, b1, as if y_i assigned to its component
        for h in setdiff(1:model.H, Si)
            d_tmp = deepcopy(D_vec[h][i,:])
            Λ_tmp_vec[ h ] = PDMat( Λ_tmp_vec[ h ] + d_tmp * d_tmp' )
            β1star00_tmp_vec[ h ] += (d_tmp .* model.y[i])
            β1star_tmp_vec[ h ] = Λ_tmp_vec[h] \ β1star00_tmp_vec[h]
            b1_tmp_vec[ h ] = get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, vcat(model.y[indx_h_vec[h]], model.y[i]), βΛβ0star_ηy, 
                Λ_tmp_vec[h], β1star_tmp_vec[h]) # may concatenate y since that argument only calulates y'y
            lmargy_in[ h ] = lmargy(Λ_tmp_vec[h], a1_vec[h] + 0.5, b1_tmp_vec[h])
        end

        # now lmargy_in and all tmp[1:H \ Si] are are as though y_i were assigned to its h slot
        # lmargy_out is still as though y_i didn't exist

        lw = [ sum(lmargy_out[ 1:end .!= h ]) + lmargy_in[h] for h = 1:model.H ]

        lw += model.state.lω
        lw += model.state.lNX[i,:]

        lw .-= maximum(lw)
        w = exp.(lw)

        ## Metropolized discrete Gibbs (Liu, 1996)
        w_cand = deepcopy(w)
        w_cand[Si] = 0.0
        cand = sample(StatsBase.Weights(w_cand))
        lar = logsumexp( lw[ 1:end .!= Si ] ) - logsumexp( lw[ 1:end .!= cand ] )
        if log(rand()) < lar
            # update the running stats
            indx_h_vec[ Si ] = setdiff( indx_h_vec[ Si ], i ) # take i out of its currently assigned set
            push!( indx_h_vec[ cand ], i ) # and add it to its newly assigned set (order doesn't matter)
            n_h_vec[ Si ] -= 1
            n_h_vec[ cand ] += 1
            a1_vec[ Si ] -= 0.5
            a1_vec[ cand ] += 0.5
            Λ1star_ηy_vec[ Si ] = deepcopy( Λ_tmp_vec[ Si ] )
            Λ1star_ηy_vec[ cand ] = deepcopy( Λ_tmp_vec[ cand ] )
            β1star00_vec[ Si ] = deepcopy( β1star00_tmp_vec[ Si ] )
            β1star00_vec[ cand ] = deepcopy( β1star00_tmp_vec[ cand ] )
            β1star_ηy_vec[ Si ] = deepcopy( β1star_tmp_vec[ Si ] )
            β1star_ηy_vec[ cand ] = deepcopy( β1star_tmp_vec[ cand ] )
            b1_vec[ Si ] = deepcopy( b1_tmp_vec[ Si ] )
            b1_vec[ cand ] = deepcopy( b1_tmp_vec[ cand ] )

            lmargy_out[ cand ] = deepcopy( lmargy_in[ cand ] ) # at the start of each iteration of the loop, lmargy_out reflects the current lmargy vector

            # update S[i]
            model.state.S[i] = cand
        else
            lmargy_out[ Si ] = deepcopy( lmargy_in[ cand ] ) # put y_i back in so lmargy_out reflects the lmargy vector before this iteration of the loop
            # nothing else?
        end

    end
    
    model.state.n_occup = length(unique(model.state.S))

    return nothing
end

