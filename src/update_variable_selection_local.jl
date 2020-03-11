# update_variable_selection_local.jl

function ldens_y_h(model::Model_BNP_WMReg_Joint, γ_h::BitArray{1}, h::Int)

    indx_h = findall(model.state.S .== h)
    n_h = length(indx_h)

    lNy = zeros(typeof(model.y[1]), n_h)
    γcand_indx = findall(γ_h)

    for i = 1:n_h
        tt = deepcopy(indx_h[i])
        μ = deepcopy(model.state.μ_y[h])
        for k in γcand_indx
            μ -= model.state.β_y[h, k] * (model.X[tt,k] - model.state.μ_x[h, k])
        end
        lNy[i] += logpdf(Normal(μ, sqrt(model.state.δ_y[h])), model.y[tt])
    end

    return lNy
end

## Calculate lNX and lωNXvec under different variable selection methods
function lNXmat_lωNXvec_h(model::Model_BNP_WMReg_Joint, γ_h::BitArray{1}, h::Int)

    lNX = deepcopy(model.state.lNX)

    γindx = findall(γ_h)
    nγ = length(γindx)

    if model.state.γδc == Inf # subset method for variable selection

        if nγ == 0

            lNX[:,h] = zeros(Float64, model.n)

        elseif nγ == 1

            lNX[:,h] = logpdf.(Normal(model.state.μ_x[h,γindx[1]], sqrt(model.state.δ_x[h,γindx[1]])), vec(model.X[:,γindx]))

        elseif nγ > 1 && model.Σx_type == :full

            βγ_x_h, δγ_x_h = βδ_x_h_modify_γ( [ model.state.β_x[j][h,:] for j = 1:length(model.state.β_x) ],
                model.state.δ_x[h,:], γ_h, model.state.γδc) # either variance-inflated or subset

            lNX[:,h] = lNX_sqfChol(Matrix(model.X[:,γindx]'), model.state.μ_x[h,γindx], βγ_x_h, δγ_x_h, true)

        elseif nγ > 1 && model.Σx_type == :diag

            lNX[:,h] = lNX_Σdiag( Matrix(model.X[:,γindx]'),
                model.state.μ_x[h,γindx], model.state.δ_x[h,γindx] )

        end

    elseif isnothing(model.state.γδc) # integration method for variable selection

        if nγ == 0

            lNX[:,h] = zeros(Float64, model.n)

        elseif nγ == 1

            if model.K > 1 && model.Σx_type == :full
                σ2x = sqfChol_to_Σ( [ model.state.β_x[k][h,:] for k = 1:length(model.state.β_x) ], model.state.δ_x[h,:] ).mat[γindx, γindx][1]
            elseif model.K > 1 && model.Σx_type == :diag
                σ2x = deepcopy(model.state.δ_x[h,γindx[1]])
            elseif model.K == 1
                σ2x = deepcopy(model.state.δ_x[h,1])
            end

            lNX[:,h] = logpdf.(Normal(model.state.μ_x[h,γindx[1]], sqrt(σ2x)), vec(model.X[:,γindx]))

        elseif nγ > 1 && model.Σx_type == :full

            Σx = PDMat( sqfChol_to_Σ( [ model.state.β_x[k][h,:] for k = 1:length(model.state.β_x) ],
                model.state.δ_x[h,:] ).mat[γindx, γindx] )

            lNX[:,h] = logpdf( MultivariateNormal(model.state.μ_x[h,γindx], Σx), Matrix(model.X[:,γindx]') )

        elseif nγ > 1 && model.Σx_type == :diag

            lNX[:,h] = lNX_Σdiag( Matrix(model.X[:,γindx]'),
                model.state.μ_x[h,γindx], model.state.δ_x[h,γindx] )

        end

    else # variance-inflation method for variable selection not supported
        throw("Variance-inflation method for local variable selection not supported.")
    end

    lωNX_vec = lωNXvec(model.state.lω, lNX)

    return lNX, lωNX_vec
end


function lNXmat_lωNXvec!(model::Model_BNP_WMReg_Joint, γ::BitArray{2})

    for h = 1:model.H
        model.state.lNX, model.state.lωNX_vec = lNXmat_lωNXvec_h(model, γ[h,:], h)
    end

    return nothing
end


# function update_γ_hk!(model::Model_BNP_WMReg_Joint, lNy_h_old::Array{T,1}, h::Int, k::Int) where T <: Real

#     γ_h_alt = deepcopy(model.state.γ[h,:])
#     γ_h_alt[k] = !γ_h_alt[k]

#     lNy_h_alt = ldens_y_h(model, γ_h_alt, h)
#     lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec_h(model, γ_h_alt, h)

#     indx_h = findall(model.state.S .== h)

#     lNx_h_alt = deepcopy( lNX_alt[indx_h, h] )
#     lNx_h_old = deepcopy( model.state.lNX[indx_h, h] )

#     la = log(model.state.π_γ[k])
#     lb = log(1.0 - model.state.π_γ[k])

#     if model.state.γ[h,k]
#         la +=  sum( lNy_h_old ) +  sum( lNx_h_old ) - sum( model.state.lωNX_vec )
#         lb +=  sum( lNy_h_alt ) + sum( lNx_h_alt ) - sum( lωNX_vec_alt )
#         ldenom = BayesInference.logsumexp([la, lb])
#         lprob_switch = lb - ldenom
#         lfc_on = la - ldenom
#     else
#         la += sum( lNy_h_alt ) + sum( lNx_h_alt ) - sum( lωNX_vec_alt )
#         lb += sum( lNy_h_old ) + sum( lNx_h_old ) - sum( model.state.lωNX_vec )
#         lprob_switch = la - BayesInference.logsumexp([la, lb])
#         lfc_on = deepcopy(lprob_switch)
#     end

#     switch = log(rand()) < lprob_switch

#     if switch
#         model.state.γ[h,k] = !model.state.γ[h,k]
#         model.state.lNX = deepcopy(lNX_alt)
#         model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
#         lNy_h_old = deepcopy(lNy_h_alt)
#     end

#     return lfc_on
# end

# function update_γ_local!(model::Model_BNP_WMReg_Joint)

#     lfc_on = zeros(Float64, model.H, model.K)

#     up_indx = findall( [ model.state.π_γ[k] < 1.0 && model.state.π_γ[k] > 0.0 for k = 1:model.K ] )

#     for h = 1:model.H

#         ## calculate lNy
#         lNy_h = ldens_y_h(model, model.state.γ[h,:], h)

#         ## loop through k
#         for k in up_indx
#             lfc_on[h,k] = update_γ_hk!(model, lNy_h, h, k)
#         end

#     end

#     return lfc_on
# end


## eta_y marginalized and Metropolized update
function update_γ_h_block!(model::Model_BNP_WMReg_Joint, up_indx::Array{Int,1}, h::Int) where T <: Real

    ## Propose deterministic switch of uniformly selected indices
    K_upd = min(3, length(up_indx))
    k_upd = sample(StatsBase.Weights( (0.5).^collect(1:K_upd) ))
    switch_indx = sample(up_indx, k_upd, replace=false)

    γ_h_alt = deepcopy(model.state.γ[h,:])
    γ_h_alt[switch_indx] = .!γ_h_alt[switch_indx]

    indx_h = findall(model.state.S .== h)
    n_h = length(indx_h)

    ## Calculate lmargy under both scenarios
    ## Calculate Lam1, a1, b1 for all h
    if n_h > 0
        Λβ0star_ηy = model.state.Λ0star_ηy * model.state.β0star_ηy
        βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)

        D_h = construct_Dh(h, model.X[indx_h,:], model.state.μ_x[h,:], model.state.γ[h,:] )
        a1_h = 0.5 .* ( model.state.ν_δy + n_h )
        Λ1star_ηy_h = get_Λ1star_ηy_h(D_h, model.state.Λ0star_ηy)
        β1star00_h = (Λβ0star_ηy + D_h'model.y[indx_h])
        β1star_ηy_h = Λ1star_ηy_h \ β1star00_h
        b1_h = get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, model.y[indx_h], βΛβ0star_ηy, Λ1star_ηy_h, β1star_ηy_h)

        lmargy_old = lmargy(Λ1star_ηy_h, a1_h, b1_h)

        D_alt_h = construct_Dh(h, model.X[indx_h,:], model.state.μ_x[h,:], γ_h_alt ) # could be more efficient, but, meh.
        Λ1star_ηy_alt_h = get_Λ1star_ηy_h(D_alt_h, model.state.Λ0star_ηy)
        β1star00_alt_h = (Λβ0star_ηy + D_alt_h'model.y[indx_h])
        β1star_ηy_alt_h = Λ1star_ηy_alt_h \ β1star00_alt_h
        b1_alt_h = get_b1_δy_h(model.state.ν_δy, model.state.s0_δy, model.y[indx_h], βΛβ0star_ηy, Λ1star_ηy_alt_h, β1star_ηy_alt_h)

        lmargy_alt = lmargy(Λ1star_ηy_alt_h, a1_h, b1_alt_h)

    else
        lmargy_old = 0.0
        lmargy_alt = 0.0
    end

    ## Calculate lNX under alternate scenario
    lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec_h(model, γ_h_alt, h)

    ## Calculate lNx under both scenarios
    lNx_h_alt = deepcopy( lNX_alt[indx_h, h] )
    lNx_h_old = deepcopy( model.state.lNX[indx_h, h] )

    ## Metropolis step
    lp_old = sum( log.(model.state.π_γ[switch_indx][ findall( model.state.γ[h, switch_indx] ) ]) ) +
             sum( log.( 1.0 .- model.state.π_γ[switch_indx][ findall( .!model.state.γ[h, switch_indx] ) ]) )
    lp_old += (sum(lNx_h_old) - sum(model.state.lωNX_vec) + sum(lmargy_old) )

    lp_alt = sum( log.(model.state.π_γ[switch_indx][ findall( γ_h_alt[switch_indx] ) ]) ) +
             sum( log.( 1.0 .- model.state.π_γ[switch_indx][ findall( .!γ_h_alt[switch_indx] ) ]) )
    lp_alt += (sum(lNx_h_alt) - sum(lωNX_vec_alt) + sum(lmargy_alt) )

    switch = log(rand()) < ( lp_alt - lp_old )

    if switch
        model.state.γ[h,:] = γ_h_alt
        model.state.lNX = deepcopy(lNX_alt)
        model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
    end

    return nothing
end


# # Metropolized update only
# function update_γ_h_block!(model::Model_BNP_WMReg_Joint, up_indx::Array{Int,1}, h::Int) where T <: Real

#     ## Propose deterministic switch of uniformly selected indices
#     K_upd = min(3, length(up_indx)) # maximum possible gammas to update
#     k_upd = sample(StatsBase.Weights( (0.5).^collect(1:K_upd) )) # select how many to update
#     switch_indx = sample(up_indx, k_upd, replace=false)

#     γ_h_alt = deepcopy(model.state.γ[h,:])
#     γ_h_alt[switch_indx] = .!γ_h_alt[switch_indx]

#     indx_h = findall(model.state.S .== h)
#     n_h = length(indx_h)

#     ## Calculate lNy under both scenarios
#     lNy_h_alt = ldens_y_h(model, γ_h_alt, h)
#     lNy_h_old = ldens_y_h(model, model.state.γ[h,:], h)

#     ## Calculate lNX under alternate scenario
#     lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec_h(model, γ_h_alt, h)

#     ## Calculate lNx under both scenarios
#     lNx_h_alt = deepcopy( lNX_alt[indx_h, h] )
#     lNx_h_old = deepcopy( model.state.lNX[indx_h, h] )

#     ## Metropolis step
#     lp_old = sum( log.(model.state.π_γ[switch_indx][ findall( model.state.γ[h, switch_indx] ) ]) ) +
#              sum( log.( 1.0 .- model.state.π_γ[switch_indx][ findall( .!model.state.γ[h, switch_indx] ) ]) )
#     lp_old += ( sum(lNx_h_old) - sum(model.state.lωNX_vec) + sum( lNy_h_old ) )

#     lp_alt = sum( log.(model.state.π_γ[switch_indx][ findall( γ_h_alt[switch_indx] ) ]) ) +
#              sum( log.( 1.0 .- model.state.π_γ[switch_indx][ findall( .!γ_h_alt[switch_indx] ) ]) )
#     lp_alt += ( sum(lNx_h_alt) - sum(lωNX_vec_alt) + sum( lNy_h_alt ) )

#     switch = log(rand()) < ( lp_alt - lp_old )

#     if switch
#         model.state.γ[h,:] = γ_h_alt
#         model.state.lNX = deepcopy(lNX_alt)
#         model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
#     end

#     return nothing
# end





function update_γ_local!(model::Model_BNP_WMReg_Joint)

    up_indx = findall( [ model.state.π_γ[k] < 1.0 && model.state.π_γ[k] > 0.0 for k = 1:model.K ] )

    if length(up_indx) > 0
        for h = 1:model.H

            update_γ_h_block!(model, up_indx, h)

        end
    end
    return zeros(model.H, model.K)
end

function update_π_γ!(model::Model_BNP_WMReg_Joint)

    for k = 1:model.K

        if model.state.ξ[k]

            nn = sum( model.state.γ[:,k] )
            aa = model.prior.π_sh[k,1] + nn
            bb = model.prior.π_sh[k,2] + model.H - nn

            model.state.π_γ[k] = rand( Beta( aa , bb ) )

        else

            model.state.π_γ[k] = 0.0

        end

    end

    return nothing
end

function update_ξ!(model::Model_BNP_WMReg_Joint) # full conditional probabilities could be pre-computed

    if typeof(model.prior.π_ξ) == Float64
        π_ξ = fill(model.prior.π_ξ, model.K)
    elseif typeof(model.prior.π_ξ) == Array{Float64, 1}
        π_ξ = deepcopy(model.prior.π_ξ)
    end

    for k = 1:model.K

        nn = sum( model.state.γ[:,k] )

        if nn > 0

            model.state.ξ[k] = true

        else

            laa0 = logabsgamma( model.prior.π_sh[k,2] + model.H )[1] + logabsgamma( model.prior.π_sh[k,1] + model.prior.π_sh[k,2] )[1] -
                logabsgamma( model.prior.π_sh[k,2] )[1] - logabsgamma( model.prior.π_sh[k,1] + model.prior.π_sh[k,2] + model.H )[1]
            aa0 = exp(laa0)

            aa = π_ξ[k] * aa0
            p1 = aa / ( aa + 1.0 - π_ξ[k] )

            model.state.ξ[k] = rand() < p1

        end

    end

    return nothing
end
