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
                σ2x = deepcopy(model.state.δ_x[h,γindx])
            elseif model.K == 1
                σ2x = deepcopy(model.state.δ_x[h,1])
            end

            lNX[:,h] = logpdf.(Normal(model.state.μ_x[h,γindx], sqrt(σ2x)), vec(model.X[:,γindx]))

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


function update_γ_hk!(model::Model_BNP_WMReg_Joint, lNy_h_old::Array{T,1}, h::Int, k::Int) where T <: Real

    γ_h_alt = deepcopy(model.state.γ[h,:])
    γ_h_alt[k] = !γ_h_alt[k]

    lNy_h_alt = ldens_y_h(model, γ_h_alt, h)
    lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec_h(model, γ_h_alt, h)

    indx_h = findall(model.state.S .== h)

    lNx_h_alt = deepcopy( lNX_alt[indx_h, h] )
    lNx_h_old = deepcopy( model.state.lNX[indx_h, h] )

    la = log(model.state.π_γ[k])
    lb = log(1.0 - model.state.π_γ[k])

    if model.state.γ[h,k]
        la +=  sum( lNy_h_old ) +  sum( lNx_h_old ) - sum( model.state.lωNX_vec )
        lb +=  sum( lNy_h_alt ) + sum( lNx_h_alt ) - sum( lωNX_vec_alt )
        ldenom = BayesInference.logsumexp([la, lb])
        lprob_switch = lb - ldenom
        lfc_on = la - ldenom
    else
        la += sum( lNy_h_alt ) + sum( lNx_h_alt ) - sum( lωNX_vec_alt )
        lb += sum( lNy_h_old ) + sum( lNx_h_old ) - sum( model.state.lωNX_vec )
        lprob_switch = la - BayesInference.logsumexp([la, lb])
        lfc_on = deepcopy(lprob_switch)
    end

    switch = log(rand()) < lprob_switch

    if switch
        model.state.γ[h,k] = !model.state.γ[h,k]
        model.state.lNX = deepcopy(lNX_alt)
        model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
        lNy_h_old = deepcopy(lNy_h_alt)
    end

    return lfc_on
end

function update_γ_local!(model::Model_BNP_WMReg_Joint)

    lfc_on = zeros(Float64, model.H, model.K)

    for h = 1:model.H

        ## calculate lNy
        lNy_h = ldens_y_h(model, model.state.γ[h,:], h)

        up_indx = findall( [ model.state.π_γ[k] < 1.0 && model.state.π_γ[k] > 0.0 for k = 1:model.K ] )
        
        ## loop through k
        for k in up_indx
            lfc_on[h,k] = update_γ_hk!(model, lNy_h, h, k)
        end

    end

    return lfc_on
end
