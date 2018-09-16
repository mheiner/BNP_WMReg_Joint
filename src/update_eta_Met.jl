# update_eta_Met.jl

## not fully compatible with K=1



# pre_compute Λ0_ηy*β0_ηy and β0_ηy'Λ0_ηy*β0_ηy which are not indexed by h
function update_η_h_Met!(model::Mod_DPmRegJoint, h::Int, Λβ0_ηy::Array{T,1}, βΛβ0_ηy::T) where T <: Real

    indx_h = findall(model.state.S.==h)
    n_h = length(indx_h)

    y_h = model.y[indx_h] # doesn't need to copy if indexing (result of call to getindex)
    X_h = model.X[indx_h,:]

    β_x_h_old = [ model.state.β_x[k][h,:] for k = 1:model.K ]
    lδ_x_h_old = log.(model.state.δ_x[h,:])
    ηlδ_x_old = vcat(model.state.μ_x[h,:],
                   vcat(β_x_h_old...),
                   lδ_x_h_old)

    ## Draw candidate for η_x
    ηlδ_x_cand = ηlδ_x_old + rand(MvNormal(model.cSig_ηlδx[h])))
    μ_x_h_cand = ηlδ_x_cand[model.indx_ηx[:μ]]
    β_x_h_candvec = ηlδ_x_cand[model.indx_ηx[:β]]
    β_x_h_cand = [ β_x_h_candvec[model.indx_β_x[k]] for k = 1:(model.K-1) ]
    lδ_x_h_cand = ηlδ_x_cand[model.indx_ηx[:δ]]
    δ_x_h_cand = exp.(lδ_x_h_cand)

    if n_h == 0

        ## Metropolis step for η_x

            ## Compute lωNX_vec for candidate
            lNX_cand = lNXmat( model.X, μ_x_h_cand, β_x_h_cand, δ_x_h_cand )
            lωNX_vec_cand = lωNXvec(model.state.lω, lNX_cand) # n vector

            ## Compute acceptance ratio
            lar = lG0_ηx(μ_x_h_cand, β_x_h_cand, lδ_x_h_cand, model.state) -
                    sum(lωNX_vec_cand) -
                    lG0_ηx(model.state.μ_x[h,:], β_x_h_old, lδ_x_h_old, model.state) +
                    sum(model.state.lωNX_vec)

            ## Decision and update
            if log(rand()) < lar # accept

                model.state.μ_x[h,:] = μ_x_h_cand
                for k = 1:(model.K - 1)
                    model.state.β_x[k][h,:] = β_x_h_cand[k]
                end
                model.state.δ_x[h,:] = δ_x_h_cand

                model.state.accpt[h] += 1

                ## G0 draw η_y ( use prec. prameterization )


            else # reject


                ## Full conditional draw for η_y



            end

    else

        ## Metropolis step for η_x

            ## Important quantities
            D_h = construct_Dh(h, X_h, model.state.μ_x[h,:])

            Λ1_ηy_h = PDMat(D_h'D_h + model.state.Λ0_ηy)
            β1_ηy_h = Λ1_ηy_h \ (Λβ0_ηy + D_h'y_h)

            a1_δy = (model.state.ν_δy + n_h) / 2.0 # posterior IG shape
            b1_δy = 0.5 * (model.state.ν_δy * model.state.s0_δy +
                                y_h'y_h + βΛβ0_ηy - β1_ηy_h'Λ1_ηy_h*β1_ηy_h ) # posterior IG scale

            ## Compute lωNX_vec, other quantities for candidate


            ## Compute acceptance ratio
            lar = lcc_ηx() - lcc_ηx()

            ## Decision and update
            if log(rand()) < lar # accept

                model.state.μ_x[h,:] = μ_x_h_cand
                for k = 1:(model.K - 1)
                    model.state.β_x[k][h,:] = β_x_h_cand[k]
                end
                model.state.δ_x[h,:] = δ_x_h_cand

                ## Full conditional draw for η_y ( use prec. prameterization )


            else # reject
                ## Full conditional draw for η_y



            end
    end

    ### REDO THIS TO JUST KEEP SAMPLES DURING ADAPTING
    if model.state.adapt
        model.state.accpt_iter += 1
        model.state.runningsum_ηx += ηlδ_x_cand
        runningmean = model.state.runningsum_ηx / float(model.state.accpt_iter)
        runningdev = ( ηlδ_x_cand - runningmean )
        model.state.runningSS_ηx = runningdev * runningdev'
    end

    return nothing
end


function construct_Dh(h::Int, Xh::Array{T, 2}, μ_x_h::Array{T, 1}) where T <: Real
    D0 = broadcast(-, μ_x_h, Xh')'
    return hcat( ones(size(Xh,1)), D0 )
end

function lG0_ηx(μ::Array{T, 1}, β_x::Array{Array{T, 1}, 1}, lδ::Array{T, 1}, model.state) where T <: Real
    Q3_μ = - 0.5 * PDMats.quad( model.state.Λ0_μx, (μ - model.state.μ0_μx) )
    Q3_β = 0.0
    Q3_δ = 0.0
    for k = 1:(model.K - 1)
        Q3_β -= 0.5 * PDMats.quad( model.state.Λ0_βx, (β_x[k] - model.state.β0_βx[k]) )
        Q3_δ -= ( (0.5*model.state.ν_δx[k] + 1.0)*lδ[k] +
                    0.5*model.state.ν_δx[k]*model.state.s0_δx[k]/exp(lδ[k]) )
    end
    Q3_δ -= ( (0.5*model.state.ν_δx[model.K] + 1.0)*lδ[model.K] +
                0.5*model.state.ν_δx[model.K]*model.state.s0_δx[model.K]/exp(lδ[model.K]) )

    Q3 = Q3_μ + Q3_β + Q3_δ
    return Q3
end


### log-collapsed conditional
function lcc_ηx()

    # Q1


    # Q2


    # Q3
    Q3 = lG0_ηx(μ, β_x, δ, model.state)

    # Q4


    # Q5


    # Q6 # Jacobian

    return Q1 - Q2 + Q3 - 0.5*Q4 - (a1_δy + n_h / 2.0)*Q5 + Q6

end
