# update_variable_selection.jl

export βδ_x_modify_γ, δ_x_modify_γ, lNXmat_lωNXvec;

function ldens_y(model::Model_DPmRegJoint, γ_cand::BitArray{1})

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
function βδ_x_modify_γ(β_x::Array{Array{T, 2}, 1}, δ_x::Array{T, 2},
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
function βδ_x_modify_γ(β_x::Array{Array{T, 2}, 1}, δ_x::Array{T, 2},
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
        βout[:,k] *= 0
    end
    return βout
end


## Calculate lNX and lωNXvec under different variable selection methods
function lNXmat_lωNXvec(model::Model_DPmRegJoint, γ::BitArray{1})

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
        elseif nγ > 1
            βγ_x, δγ_x = βδ_x_modify_γ(model.state.β_x, model.state.δ_x,
                                       γ, model.state.γδc)
            lNX = lNXmat(model.X[:,γindx],
                         model.state.μ_x[:,γindx], βγ_x, δγ_x) # n by H matrix
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        end

    elseif model.state.γδc == nothing # integration method for variable selection

        if nγ == 0
            lNX = zeros(Float64, model.n, model.H)
            lωNX_vec = zeros(Float64, model.n)
        elseif nγ == 1
            σ2xs = [ sqfChol_to_Σ( [ model.state.β_x[k][h,:] for k = 1:(model.K-1) ], model.state.δ_x[h,:] ).mat[γindx, γindx][1] for h = 1:model.H ]
            lNX = lNXmat(model.X[:,γindx], model.state.μ_x[:,γindx], σ2xs)
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        elseif nγ > 1
            Σxs = [ PDMat( sqfChol_to_Σ( [ model.state.β_x[k][h,:] for k = 1:(model.K-1) ], model.state.δ_x[h,:] ).mat[γindx, γindx] ) for h = 1:model.H ]
            lNX = hcat( [ logpdf(MultivariateNormal(model.state.μ_x[h,γindx], Σxs[h]), Matrix(model.X[:,γindx]')) for h = 1:model.H ]... ) # n by H matrix
            lωNX_vec = lωNXvec(model.state.lω, lNX)
        end

    else # variance-inflation method for variable selection

        if model.K > 1
            βγ_x, δγ_x = βδ_x_modify_γ(model.state.β_x, model.state.δ_x,
                                       γ, model.state.γδc)
            lNX = lNXmat(model.X, model.state.μ_x, βγ_x, δγ_x)
        else
            δγ_x = δ_x_modify_γ(model.state.δ_x, γ, model.state.γδc)
            lNX = lNXmat(vec(model.X), vec(model.state.μ_x), vec(δγ_x))
        end
        lωNX_vec = lωNXvec(model.state.lω, lNX)
    end

    return lNX, lωNX_vec
end



function update_γ_k!(model::Model_DPmRegJoint, lNy_old::Array{T,1}, k::Int) where T <: Real

    γ_alt = deepcopy(model.state.γ)
    γ_alt[k] = !γ_alt[k]

    lNy_alt = ldens_y(model, γ_alt)
    lNX_alt, lωNX_vec_alt = lNXmat_lωNXvec(model, γ_alt)

    lNx_alt = [ deepcopy(lNX_alt[i, model.state.S[i]]) for i = 1:model.n ]
    lNx_old = [ deepcopy(model.state.lNX[i, model.state.S[i]]) for i = 1:model.n ]

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
