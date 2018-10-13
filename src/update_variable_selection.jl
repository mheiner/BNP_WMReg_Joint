# update_variable_selection.jl

function ldens_y(model::Model_DPmRegJoint, γ_cand::BitArray{1})

    lNy = zeros(typeof(model.y[1]), model.n)
    γcand_indx = findall(γ_cand) # currently assumes global γ

    for i = 1:model.n
        μ = model.state.μ_y[model.state.S[i]]
        for k in γcand_indx
            μ -= β_y[model.state.S[i], k] * (model.state.μ_x[model.state.S[i], k] - model.X[i,k])
        end
        lNy[i] += logpdf(Normal(μ, sqrt(model.state.δ_y[model.state.S[i]])), model.y[i])
    end

    return lNy
end

## single component versions of these functions are found in update_eta_Met.jl
function βδ_x_modify_γ(β_x::Array{Array{T, 2}, 1}, δ_x::Array{T, 2},
                        γ::BitArray{1}, γδc::Array{T, 1}) where T <: Real
    modify_indx = findall(.!(γ))
    K = length(γ)

    βout = deepcopy(β_x) # vector of H by (k in (K-1):1) matrices
    δout = deepcopy(δ_x) # H by K matrix

    for k in modify_indx # this works even if no modifications are necessary
        δout[:,k] += γδc[k]
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
        δout[:,k] *= γδc[k]
    end
    return δout
end



function update_γ_k!(model::Model_DPmRegJoint, lNy_old::Array{T,1}, k::Int) where T <: Real

    γ_alt = copy(model.state.γ)
    γ_alt[k] = !γ_alt[k]

    lNy_alt = ldens_y(model, γ_alt)

    if model.K > 1
        βγ_x_alt, δγ_x_alt = βδ_x_modify_γ(model.state.β_x, model.state.δ_x,
                                           γ_alt, model.state.γδc)

        lNX_alt = lNXmat(model.X, model.state.μ_x, βγ_x_alt, δγ_x_alt) # n by H matrix
    else
        δγ_x_alt = δ_x_modify_γ(model.state.δ_x, γ_alt, model.state.γδc)

        lNX_alt = lNXmat(model.X, model.state.μ_x, δγ_x_alt) # n by H matrix
    end

    lNx_alt = [ copy(lNX_alt[i, model.state.S[i]]) for i = 1:model.n ]
    lNx_old = [ copy(model.state.lNX[i, model.state.S[i]]) for i = 1:model.n ]

    lωNX_vec_alt = lωNXvec(model.state.lω, lNX_alt)


    la = log(model.state.π_γ[k])
    lb = log(1.0 - model.state.π_γ[k])

    if model.state.γ[k]
        la +=  sum( lNy_old + lNx_old - model.state.lωNX_vec )
        lb +=  sum( lNy_alt + lNx_alt - lωNX_vec_alt )
        lprob_switch = lb - BayesInference.logsumexp([la, lb])
    else
        la += sum( lNy_alt + lNx_alt - lωNX_vec_alt )
        lb += sum( lNy_old + lNx_old - model.state.lωNX_vec )
        lprob_switch = la - BayesInference.logsumexp([la, lb])
    end

    switch = log(rand()) < lprob_switch

    if switch
        model.state.γ[k] = !model.state.γ[k]
        model.state.lNX = deepcopy(lNX_alt)
        model.state.lωNX_vec = deepcopy(lωNX_vec_alt)
        lNy_old = deepcopy(lNy_alt)
    end

    return nothing
end

function update_γ!(model::Model_DPmRegJoint)

    ## calculate lNy
    lNy = ldens_y(model, model.state.γ)

    ## loop through k
    for k = 1:model.K
        update_γ_k!(model, lNy, k)
    end

    return nothing
end
