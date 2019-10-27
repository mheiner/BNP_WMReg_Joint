# update_variable_selection_global.jl

export βδ_x_modify_γ, δ_x_modify_γ, lNXmat_lωNXvec;

function ldens_y(model::Model_BNP_WMReg_Joint, γ_cand::BitArray{1})

    lNy = zeros(typeof(model.y[1]), model.n)
    γcand_indx = findall(γ_cand) # currently assumes global γ

    for i = 1:model.n
        μ = deepcopy(model.state.μ_y[model.state.S[i]])
        for k in γcand_indx
            μ -= model.state.β_y[model.state.S[i], k] * (model.X[i,k] - model.state.μ_x[model.state.S[i], k])
        end
        lNy[i] += logpdf(Normal(μ, sqrt(model.state.δ_y[model.state.S[i]])), model.y[i])
    end

    return lNy
end

## single component versions of these functions are found in update_eta_Met.jl
## This is the version that inflates variances
function βδ_x_modify_γ(β_x::Union{Array{Array{T, 2}, 1}, Array{Any,1}}, δ_x::Array{T, 2},
                        γ::BitArray{1}, γδc::Array{T, 1}) where T <: Real
    modify_indx = findall(.!(γ))
    K = length(γ)

    βout = deepcopy(β_x) # vector of H by (k in (K-1):1) matrices
    δout = deepcopy(δ_x) # H by K matrix

    for k in modify_indx # this works even if no modifications are necessary
        δout[:,k] .+= γδc[k]
    end

    for k = 1:(K-1)
        if !γ[k]
            βout[k] *= 0.0
        else
            modify2 = intersect((k+1):K, modify_indx)
            βout[k][:,(modify2 .- k)] *= 0.0
        end
    end

    return βout, δout
end
function δ_x_modify_γ(δ_x::Array{T, 2},
                      γ::BitArray{1}, γδc::Array{T, 1}) where T <: Real
    modify_indx = findall(.!(γ))
    δout = deepcopy(δ_x)

    for k in modify_indx # this works even if no modifications are necessary
        δout[:,k] .+= γδc[k]
    end
    return δout
end


## single component versions of these functions are found in update_eta_Met.jl
## This is the version that subsets
function βδ_x_modify_γ(β_x::Union{Array{Array{T, 2}, 1}, Array{Any,1}}, δ_x::Array{T, 2},
                        γ::BitArray{1}, γδc::Float64) where T <: Real

    γδc == Inf || throw("A single variance inflation should be equal to Inf")

    γindx = findall(γ)
    nγ = length(γindx)
    nγ > 1 || throw("βδ_x_modify_γ requires more than one selected variable.")

    βout = [ deepcopy(β_x[γindx[k]][:, (γindx[(k+1):nγ] .- γindx[k]) ])  for k = 1:(nγ-1) ] # vector of H by (nγ-1):1 matrices
    δout = deepcopy(δ_x[:,γindx]) # H nows and sum(gamma) cols

    return βout, δout
end
function δ_x_modify_γ(δ_x::Array{T, 2},
                      γ::BitArray{1}, γδc::Float64) where T <: Real
    γδc == Inf || throw("A single variance inflation should be equal to Inf")
    γindx = findall(γ)
    δout = deepcopy(δ_x[:,γindx]) # H nows and sum(gamma) cols
    return δout
end


## Just subset this in the subset method
function β_y_modify_γ(β_y::Array{T, 2}, γ::BitArray{1}) where T <: Real
    modify_indx = findall(.!(γ))
    βout = deepcopy(β_y)

    for k in modify_indx # this works even if no modifications are necessary
        βout[:,k] *= 0.0
    end
    return βout
end
function β_y_modify_γ(β_y::Array{T, 1}, γ::BitArray{1}) where T <: Real
    modify_indx = findall(.!(γ))
    βout = deepcopy(β_y)

    for k in modify_indx # this works even if no modifications are necessary
        βout[k] *= 0.0
    end
    return βout
end


## Calculate lNX and lωNXvec under different variable selection methods
function lNXmat_lωNXvec(model::Model_BNP_WMReg_Joint, γ::BitArray{1})

    γindx = findall(γ)
    nγ = length(γindx)

    if model.state.γδc == Inf # subset method for variable selection

        if nγ == 0
            lNX = zeros(Float64, model.n, model.H)
            lωNX_vec = zeros(Float64, model.n)
        elseif nγ == 1
            lNX = lNXmat(model.X[:,γindx],
                         model.state.μ_x[:,γindx], model.state.δ_x[:,γindx])
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        elseif nγ > 1 && model.Σx_type == :full
            βγ_x, δγ_x = βδ_x_modify_γ(model.state.β_x, model.state.δ_x,
                                       γ, model.state.γδc)
            lNX = lNXmat(model.X[:,γindx],
                         model.state.μ_x[:,γindx], βγ_x, δγ_x) # n by H matrix
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        elseif nγ > 1 && model.Σx_type == :diag
            lNX = lNXmat_Σdiag(model.X[:,γindx],
                model.state.μ_x[:,γindx], model.state.δ_x[:,γindx]) # n by H matrix
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        end

    elseif isnothing(model.state.γδc) # integration method for variable selection

        if nγ == 0
            lNX = zeros(Float64, model.n, model.H)
            lωNX_vec = zeros(Float64, model.n)
        elseif nγ == 1
            if model.K > 1 && model.Σx_type == :full
                σ2xs = [ sqfChol_to_Σ( [ model.state.β_x[k][h,:] for k = 1:(model.K-1) ], model.state.δ_x[h,:] ).mat[γindx, γindx][1] for h = 1:model.H ]
            elseif model.K > 1 && model.Σx_type == :diag
                σ2xs = deepcopy(model.state.δ_x[:,γindx])
            elseif model.K == 1
                σ2xs = deepcopy(model.state.δ_x[:,1])
            end
            lNX = lNXmat(model.X[:,γindx], model.state.μ_x[:,γindx], σ2xs)
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        elseif nγ > 1 && model.Σx_type == :full
            Σxs = [ PDMat( sqfChol_to_Σ( [ model.state.β_x[k][h,:] for k = 1:(model.K-1) ], model.state.δ_x[h,:] ).mat[γindx, γindx] ) for h = 1:model.H ]
            lNX = hcat( [ logpdf(MultivariateNormal(model.state.μ_x[h,γindx], Σxs[h]), Matrix(model.X[:,γindx]')) for h = 1:model.H ]... ) # n by H matrix
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        elseif nγ > 1 && model.Σx_type == :diag
            lNX = lNXmat_Σdiag(model.X[:,γindx],
                model.state.μ_x[:,γindx], model.state.δ_x[:,γindx]) # n by H matrix
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        end

    else # variance-inflation method for variable selection

        if model.K > 1 && model.Σx_type == :full
            βγ_x, δγ_x = βδ_x_modify_γ(model.state.β_x, model.state.δ_x,
                                       γ, model.state.γδc)
            lNX = lNXmat(model.X, model.state.μ_x, βγ_x, δγ_x)
        elseif model.K == 1 && model.Σx_type == :full
            δγ_x = δ_x_modify_γ(model.state.δ_x, γ, model.state.γδc)
            lNX = lNXmat(vec(model.X), vec(model.state.μ_x), vec(δγ_x))
        end
        lωNX_vec = lωNXvec(model.state.lω, lNX)
    end

    return lNX, lωNX_vec
end



# function update_γ_k!(model::Model_BNP_WMReg_Joint, lNy_old::Array{T,1}, k::Int) where T <: Real

#     γ_alt = deepcopy(model.state.γ)
#     γ_alt[k] = !γ_alt[k]

#     lNy_alt = ldens_y(model, γ_alt)
#     lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec(model, γ_alt)

#     lNx_alt = [ deepcopy(lNX_alt[i, model.state.S[i]]) for i = 1:model.n ]
#     lNx_old = [ deepcopy(model.state.lNX[i, model.state.S[i]]) for i = 1:model.n ]

#     la = log(model.state.π_γ[k])
#     lb = log(1.0 - model.state.π_γ[k])

#     if model.state.γ[k]
#         la +=  sum( lNy_old + lNx_old - model.state.lωNX_vec )
#         lb +=  sum( lNy_alt + lNx_alt - lωNX_vec_alt )
#         ldenom = BayesInference.logsumexp([la, lb])
#         lprob_switch = lb - ldenom
#         lfc_on = la - ldenom
#     else
#         la += sum( lNy_alt + lNx_alt - lωNX_vec_alt )
#         lb += sum( lNy_old + lNx_old - model.state.lωNX_vec )
#         lprob_switch = la - BayesInference.logsumexp([la, lb])
#         lfc_on = deepcopy(lprob_switch)
#     end

#     switch = log(rand()) < lprob_switch

#     if switch
#         model.state.γ[k] = !model.state.γ[k]
#         model.state.lNX = deepcopy(lNX_alt)
#         model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
#         lNy_old = deepcopy(lNy_alt)
#     end

#     return lfc_on
# end

function update_γ_block!(model::Model_BNP_WMReg_Joint, up_indx::Array{Int,1}) where T <: Real

    ## Propose deterministic switch of uniformly selected indices
    K_upd = min(3, length(up_indx))
    k_upd = sample(StatsBase.Weights( (0.5).^collect(1:K_upd) ))
    switch_indx = sample(up_indx, k_upd, replace=false)

    γ_alt = deepcopy(model.state.γ)
    γ_alt[switch_indx] = .!γ_alt[switch_indx]

    ## Calculate lmargy under both scenarios
    Λβ0star_ηy = model.state.Λ0star_ηy * model.state.β0star_ηy
    βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)

    indx_h_vec = [ findall(model.state.S .== h) for h = 1:model.H ]
    n_h_vec = [ length(indx_h_vec[h]) for h = 1:model.H ]

    D_vec = [ construct_Dh(h, model.X[indx_h_vec[h],:], model.state.μ_x[h,:], model.state.γ ) for h = 1:model.H ]
    
    ## Calculate Lam1, a1, b1 for all h
    a1_vec = 0.5 .* ( model.state.ν_δy .+ n_h_vec )
    Λ1star_ηy_vec = [ n_h_vec[h] > 0 ? get_Λ1star_ηy_h(D_vec[h], model.state.Λ0star_ηy) : deepcopy(model.state.Λ0star_ηy) for h = 1:model.H ]
    β1star00_vec = [ n_h_vec[h] > 0 ? (Λβ0star_ηy + D_vec[h]'model.y[indx_h_vec[h]]) : deepcopy(Λβ0star_ηy) for h = 1:model.H ]
    β1star_ηy_vec = [ n_h_vec[h] > 0 ? Λ1star_ηy_vec[h] \ β1star00_vec[h] : deepcopy(model.state.β0star_ηy) for h = 1:model.H ]
    b1_vec = [ n_h_vec[h] > 0 ? get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, model.y[indx_h_vec[h]], βΛβ0star_ηy, Λ1star_ηy_vec[h], β1star_ηy_vec[h]) : 0.0  for h = 1:model.H ]

    lmargy_old = [ n_h_vec[h] > 0 ? lmargy(Λ1star_ηy_vec[h], a1_vec[h], b1_vec[h]) : 0.0 for h = 1:model.H ] # unoccupied components don't contribute

    D_alt_vec = [ construct_Dh(h, model.X[indx_h_vec[h],:], model.state.μ_x[h,:], γ_alt ) for h = 1:model.H ] # could be more efficient, but, meh.
    Λ1star_ηy_alt_vec = [ n_h_vec[h] > 0 ? get_Λ1star_ηy_h(D_alt_vec[h], model.state.Λ0star_ηy) : deepcopy(model.state.Λ0star_ηy) for h = 1:model.H ]
    β1star00_alt_vec = [ n_h_vec[h] > 0 ? (Λβ0star_ηy + D_alt_vec[h]'model.y[indx_h_vec[h]]) : deepcopy(Λβ0star_ηy) for h = 1:model.H ]
    β1star_ηy_alt_vec = [ n_h_vec[h] > 0 ? Λ1star_ηy_alt_vec[h] \ β1star00_alt_vec[h] : deepcopy(model.state.β0star_ηy) for h = 1:model.H ]
    b1_alt_vec = [ n_h_vec[h] > 0 ? get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, model.y[indx_h_vec[h]], βΛβ0star_ηy, Λ1star_ηy_alt_vec[h], β1star_ηy_alt_vec[h]) : 0.0  for h = 1:model.H ]

    lmargy_alt = [ n_h_vec[h] > 0 ? lmargy(Λ1star_ηy_alt_vec[h], a1_vec[h], b1_alt_vec[h]) : 0.0 for h = 1:model.H ]

    ## Calculate lNX under alternate scenario
    lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec(model, γ_alt)

    ## Calculate lNx under both scenarios
    lNx_alt = [ deepcopy(lNX_alt[i, model.state.S[i]]) for i = 1:model.n ]
    lNx_old = [ deepcopy(model.state.lNX[i, model.state.S[i]]) for i = 1:model.n ]

    ## Metropolis step
    lp_old = sum( log.(model.state.π_γ[ findall( model.state.γ[up_indx] ) ]) ) + sum( log.( 1.0 .- model.state.π_γ[ findall( .!model.state.γ[up_indx] ) ]) )
    lp_old += (sum(lNx_old) - sum(model.state.lωNX_vec) + sum(lmargy_old) )

    lp_alt = sum( log.(model.state.π_γ[ findall( γ_alt[up_indx] ) ]) ) + sum( log.( 1.0 .- model.state.π_γ[ findall( .!γ_alt[up_indx] ) ]) )
    lp_alt += (sum(lNx_alt) - sum(lωNX_vec_alt) + sum(lmargy_alt) )

    switch = log(rand()) < ( lp_alt - lp_old )

    if switch
        model.state.γ = γ_alt
        model.state.lNX = deepcopy(lNX_alt)
        model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
    end

    return nothing
end

# function update_γ_global!(model::Model_BNP_WMReg_Joint)

#     ## calculate lNy
#     lNy = ldens_y(model, model.state.γ)

#     up_indx = findall( [ model.state.π_γ[k] < 1.0 && model.state.π_γ[k] > 0.0 for k = 1:model.K ] )
#     lfc_on = deepcopy(model.state.π_γ)

#     ## loop through k
#     for k in up_indx
#         lfc_on[k] = update_γ_k!(model, lNy, k)
#     end

#     return lfc_on
# end

function update_γ_global!(model::Model_BNP_WMReg_Joint)

    up_indx = findall( [ model.state.π_γ[k] < 1.0 && model.state.π_γ[k] > 0.0 for k = 1:model.K ] )

    if length(up_indx) > 0
        update_γ_block!(model, up_indx)
    end

    return zeros(model.K)
end


function update_γ!(model::Model_BNP_WMReg_Joint)

    if model.γ_type == :global

        lfc_on = update_γ_global!(model)

    elseif model.γ_type == :local

        lfc_on = update_γ_local!(model)

    end

    return lfc_on
end


### Functions for Zanella & Roberts (2019) tempered Gibbs algorithm. (doesn't work embedded in larger Gibbs sampler)
# function get_lp_γ(model::Model_BNP_WMReg_Joint, up_indx::Vector{Int}, κ_γ::T) where T <: Real
#
#     n_up = length(up_indx)
#     lp_out = Vector{T}(undef, n_up)
#     lfc_on = log.(deepcopy(model.state.π_γ))
#
#     lNX_out = [ Matrix{T}(undef, model.n, model.H) for ii = 1:n_up ]
#     lωNX_vec_out = [ Vector{T}(undef, model.n) for ii = 1:n_up ]
#
#     lNy_old = ldens_y(model, model.state.γ)
#     lNx_old = [ deepcopy(model.state.lNX[i, model.state.S[i]]) for i = 1:model.n ]
#
#     lκ_γ = log(κ_γ)
#     # lκ_γ = log(1000.0)
#     ln_up = log( float(n_up) )
#     l2 = log(2.0)
#
#     for ii = 1:n_up
#         k = up_indx[ii]
#
#         γ_alt = deepcopy(model.state.γ)
#         γ_alt[k] = !γ_alt[k]
#
#         lNy_alt = ldens_y(model, γ_alt)
#         lNX_out[ii], lωNX_vec_out[ii] = lNXmat_lωNXvec(model, γ_alt)
#
#         lNx_alt = [ deepcopy(lNX_out[ii][i, model.state.S[i]]) for i = 1:model.n ]
#
#         lfc_part_on = log(model.state.π_γ[k]) # log prior probability of γ = 1
#         lfc_part_off = log(1.0 - model.state.π_γ[k])
#
#         if model.state.γ[k]
#             lfc_part_on +=  sum( lNy_old + lNx_old - model.state.lωNX_vec )
#             lfc_part_off +=  sum( lNy_alt + lNx_alt - lωNX_vec_out[ii] )
#             lfc_denom = BayesInference.logsumexp([lfc_part_on, lfc_part_off])
#             lfc_on[k] = lfc_part_on - lfc_denom
#
#             lp_numer = BayesInference.logsumexp( [ lfc_on[k], lκ_γ - ln_up ] )
#             lp_out[ii] = lp_numer - ( l2 + lfc_on[k] )
#         else
#             lfc_part_on += sum( lNy_alt + lNx_alt - lωNX_vec_out[ii] )
#             lfc_part_off += sum( lNy_old + lNx_old - model.state.lωNX_vec )
#             lfc_denom = BayesInference.logsumexp([lfc_part_on, lfc_part_off])
#             lfc_on[k] = lfc_part_on - lfc_denom
#             lfc_off = lfc_part_off - lfc_denom
#
#             lp_numer = BayesInference.logsumexp( [ lfc_on[k], lκ_γ - ln_up ] )
#             lp_out[ii] = lp_numer - ( l2 + lfc_off )
#         end
#
#     end
#
#     return lp_out, lNX_out, lωNX_vec_out, lfc_on
# end
#
# function update_γ!(model::Model_BNP_WMReg_Joint) ## Zanella & Roberts (2019) tempered Gibbs algorithm.
#
#     up_indx = findall( [ model.state.π_γ[k] < 1.0 && model.state.π_γ[k] > 0.0 for k = 1:model.K ] )
#     d = length(up_indx)
#
#     if d > 0
#         κ_γ = sum(model.state.π_γ[up_indx]) # apriori expected number of active lags (that aren't fixed on or off)
#
#         ## get weights
#         lp, lNX_alt, lωNX_vec_alt, lfc_on = get_lp_γ(model, up_indx, κ_γ)
#         lsump = BayesInference.logsumexp(lp)
#         lp .-= lsump
#
#         ## select which indicator to switch
#         switch_indx = StatsBase.sample(1:d, Weights(exp.(lp)))
#
#         ## switch the selected indicator
#         model.state.γ[up_indx[switch_indx]] = !model.state.γ[up_indx[switch_indx]]
#
#         ## assign model state quantities
#         model.state.lNX = lNX_alt[switch_indx]
#         model.state.lωNX_vec = lωNX_vec_alt[switch_indx]
#
#         ## assign sample log importance weight
#         model.state.lwimp = -lsump
#     end
#
#     return lfc_on
# end
