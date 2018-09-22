# update_eta_Met.jl

# pre_compute Λ0star_ηy*β0star_ηy and β0star_ηy'Λ0star_ηy*β0star_ηy which are not indexed by h
function update_η_h_Met!(model::Model_DPmRegJoint, h::Int, Λβ0star_ηy::Array{T,1}, βΛβ0star_ηy::T) where T <: Real

    ## if you modify this function, you must modify its K1 twin below.

    indx_h = findall(model.state.S.==h)
    n_h = length(indx_h)

    μ_x_h_old = model.state.μ_x[h,:]
    lδ_x_h_old = log.(model.state.δ_x[h,:])
    β_x_h_old = [ model.state.β_x[k][h,:] for k = 1:(model.K-1) ]
    ηlδ_x_old = vcat(μ_x_h_old,
        vcat(β_x_h_old...),
        lδ_x_h_old)

    ## Draw candidate for η_x
    ηlδ_x_cand = ηlδ_x_old + rand(MvNormal(model.state.cSig_ηlδx[h]))
    μ_x_h_cand = ηlδ_x_cand[model.indx_ηx[:μ]]
    β_x_h_candvec = ηlδ_x_cand[model.indx_ηx[:β]]
    β_x_h_cand = [ β_x_h_candvec[model.indx_β_x[k]] for k = 1:(model.K-1) ]
    lδ_x_h_cand = ηlδ_x_cand[model.indx_ηx[:δ]]
    δ_x_h_cand = exp.(lδ_x_h_cand)

    ## Compute lωNX_vec for candidate
    lNX_mat_cand = copy(model.state.lNX)
    lNX_mat_cand[:,h] = lNX_sqfChol( Matrix(model.X'), μ_x_h_cand, β_x_h_cand, δ_x_h_cand )
    lωNX_vec_cand = lωNXvec(model.state.lω, lNX_mat_cand) # n vector

    if n_h == 0

        ## Metropolis step for η_x

            ## Compute acceptance ratio
            lar = lG0_ηlδx(μ_x_h_cand, β_x_h_cand, lδ_x_h_cand, model.state) -
                sum(lωNX_vec_cand) -
                lG0_ηlδx(μ_x_h_old, β_x_h_old, lδ_x_h_old, model.state) +
                sum(model.state.lωNX_vec)

            ## Decision and update
            if log(rand()) < lar # accept

                model.state.μ_x[h,:] = μ_x_h_cand
                for k = 1:(model.K - 1)
                    model.state.β_x[k][h,:] = β_x_h_cand[k]
                end
                model.state.δ_x[h,:] = δ_x_h_cand

                model.state.lNX[:,h] = lNX_mat_cand[:,h]
                model.state.lωNX_vec = copy(lωNX_vec_cand)

                ηlδ_x_out = ηlδ_x_cand
                model.state.accpt[h] += 1

            else # reject
                ηlδ_x_out = ηlδ_x_old
            end

            ## Full conditional draw (from G0) for η_y

            model.state.δ_y[h] = rand(InverseGamma(0.5*model.state.ν_δy,
                                        0.5*model.state.ν_δy*model.state.s0_δy))
            βstar_ηy = model.state.β0star_ηy + (model.state.Λ0star_ηy.chol.U \ randn((model.K+1)) .* sqrt(model.state.δ_y[h]))
            model.state.μ_y[h] = βstar_ηy[model.indx_ηy[:μ]]
            model.state.β_y[h,:] = βstar_ηy[model.indx_ηy[:β]]

    else # some observations are assigned to cluster h (i.e., n_h > 0)

        ## Metropolis step for η_x

            ## Candidate, lNX, lωNXvec, previously computed
            ## Important quantities

            y_h = model.y[indx_h] # doesn't need to copy if indexing (result of call to getindex)
            X_h = model.X[indx_h,:]
            D_h_old = construct_Dh(h, X_h, model.state.μ_x[h,:])

            Λ1star_ηy_h_old = PDMat(D_h_old'D_h_old + model.state.Λ0star_ηy)
            β1star_ηy_h_old = Λ1star_ηy_h_old \ (Λβ0star_ηy + D_h_old'y_h)

            a1_δy_old = (model.state.ν_δy + n_h) / 2.0 # posterior IG shape
            b1_δy_old = 0.5 * (model.state.ν_δy * model.state.s0_δy +
                                y_h'y_h + βΛβ0star_ηy - PDMats.quad(Λ1star_ηy_h_old, β1star_ηy_h_old)) # posterior IG scale

            ## Important quantities for candidate
            D_h_cand = construct_Dh(h, X_h, μ_x_h_cand)

            Λ1star_ηy_h_cand = PDMat(D_h_cand'D_h_cand + model.state.Λ0star_ηy)
            β1star_ηy_h_cand = Λ1star_ηy_h_cand \ (Λβ0star_ηy + D_h_cand'y_h)

            a1_δy_cand = (model.state.ν_δy + n_h) / 2.0 # posterior IG shape
            b1_δy_cand = 0.5 * (model.state.ν_δy * model.state.s0_δy +
                                y_h'y_h + βΛβ0star_ηy - PDMats.quad(Λ1star_ηy_h_cand, β1star_ηy_h_cand) ) # posterior IG scale

            ## Compute acceptance ratio
            lar = lcc_ηlδx(h, indx_h, lNX_mat_cand, lωNX_vec_cand,
                model.state, μ_x_h_cand, β_x_h_cand, lδ_x_h_cand,
                Λ1star_ηy_h_cand, a1_δy_cand, b1_δy_cand) -
                lcc_ηlδx(h, indx_h, model.state.lNX, model.state.lωNX_vec,
                    model.state, μ_x_h_old, β_x_h_old, lδ_x_h_old,
                    Λ1star_ηy_h_old, a1_δy_old, b1_δy_old)


            ## Decision and update
            if log(rand()) < lar # accept

                model.state.μ_x[h,:] = μ_x_h_cand
                for k = 1:(model.K - 1)
                    model.state.β_x[k][h,:] = β_x_h_cand[k]
                end
                model.state.δ_x[h,:] = δ_x_h_cand

                model.state.lNX[:,h] = lNX_mat_cand[:,h]
                model.state.lωNX_vec = copy(lωNX_vec_cand)

                ηlδ_x_out = ηlδ_x_cand
                model.state.accpt[h] += 1

                ## Full conditional draw for η_y ( use prec. prameterization )

                model.state.δ_y[h] = rand(InverseGamma(a1_δy_cand, b1_δy_cand))
                βstar_ηy = β1star_ηy_h_cand + (Λ1star_ηy_h_cand.chol.U \ randn((model.K+1)) .* sqrt(model.state.δ_y[h]))
                model.state.μ_y[h] = βstar_ηy[model.indx_ηy[:μ]]
                model.state.β_y[h,:] = βstar_ηy[model.indx_ηy[:β]]

            else # reject

                ηlδ_x_out = ηlδ_x_old

                ## Full conditional draw for η_y

                model.state.δ_y[h] = rand(InverseGamma(a1_δy_old, b1_δy_old))
                βstar_ηy = β1star_ηy_h_old + (Λ1star_ηy_h_old.chol.U \ randn((model.K+1)) .* sqrt(model.state.δ_y[h]))
                model.state.μ_y[h] = βstar_ηy[model.indx_ηy[:μ]]
                model.state.β_y[h,:] = βstar_ηy[model.indx_ηy[:β]]

            end
    end

    if model.state.adapt
        model.state.adapt_iter += 1
        model.state.runningsum_ηlδx[h,:] += ηlδ_x_out
        runningmean = model.state.runningsum_ηlδx[h,:] / float(model.state.adapt_iter)
        runningdev = ( ηlδ_x_out - runningmean )
        model.state.runningSS_ηlδx[h,:,:] = runningdev * runningdev' # the mean is changing, but this approx. is fine.
    end

    return nothing
end

function update_η_h_Met_K1!(model::Model_DPmRegJoint, h::Int, Λβ0star_ηy::Array{T,1}, βΛβ0star_ηy::T) where T <: Real

    ## if you modify this function, you must modify the original above.

    indx_h = findall(model.state.S.==h)
    n_h = length(indx_h)

    μ_x_h_old = model.state.μ_x[h,:]
    lδ_x_h_old = log.(model.state.δ_x[h,:])
    ηlδ_x_old = vcat(μ_x_h_old, lδ_x_h_old)

    ## Draw candidate for η_x
    ηlδ_x_cand = ηlδ_x_old + rand(MvNormal(model.state.cSig_ηlδx[h]))
    μ_x_h_cand = ηlδ_x_cand[model.indx_ηx[:μ]]
    lδ_x_h_cand = ηlδ_x_cand[model.indx_ηx[:δ]]
    δ_x_h_cand = exp.(lδ_x_h_cand)

    ## Compute lωNX_vec for candidate
    lNX_mat_cand = copy(model.state.lNX)
    lNX_mat_cand[:,h] = logpdf.(Normal(μ_x_h_cand[1], sqrt(δ_x_h_cand[1])), vec(model.X))
    lωNX_vec_cand = lωNXvec(model.state.lω, lNX_mat_cand) # n vector

    if n_h == 0

        ## Metropolis step for η_x

            ## Compute acceptance ratio
            lar = lG0_ηlδx(μ_x_h_cand[1], lδ_x_h_cand[1], model.state) -
                sum(lωNX_vec_cand) -
                lG0_ηlδx(μ_x_h_old[1], lδ_x_h_old[1], model.state) +
                sum(model.state.lωNX_vec)

            ## Decision and update
            if log(rand()) < lar # accept

                model.state.μ_x[h,:] = μ_x_h_cand
                model.state.δ_x[h,:] = δ_x_h_cand

                model.state.lNX[:,h] = lNX_mat_cand[:,h]
                model.state.lωNX_vec = copy(lωNX_vec_cand)

                ηlδ_x_out = ηlδ_x_cand
                model.state.accpt[h] += 1

            else # reject
                ηlδ_x_out = ηlδ_x_old
            end

            ## Full conditional draw (from G0) for η_y

            model.state.δ_y[h] = rand(InverseGamma(0.5*model.state.ν_δy,
                                        0.5*model.state.ν_δy*model.state.s0_δy))
            βstar_ηy = model.state.β0star_ηy + (model.state.Λ0star_ηy.chol.U \ randn((model.K+1)) .* sqrt(model.state.δ_y[h]))
            model.state.μ_y[h] = βstar_ηy[model.indx_ηy[:μ]]
            model.state.β_y[h,:] = βstar_ηy[model.indx_ηy[:β]]

    else # some observations are assigned to cluster h (i.e., n_h > 0)

        ## Metropolis step for η_x

            ## Candidate, lNX, lωNXvec, previously computed
            ## Important quantities

            y_h = model.y[indx_h] # doesn't need to copy if indexing (result of call to getindex)
            X_h = model.X[indx_h,:]
            D_h_old = construct_Dh(h, X_h, model.state.μ_x[h,:])

            Λ1star_ηy_h_old = PDMat(D_h_old'D_h_old + model.state.Λ0star_ηy)
            β1star_ηy_h_old = Λ1star_ηy_h_old \ (Λβ0star_ηy + D_h_old'y_h)

            a1_δy_old = (model.state.ν_δy + n_h) / 2.0 # posterior IG shape
            b1_δy_old = 0.5 * (model.state.ν_δy * model.state.s0_δy +
                                y_h'y_h + βΛβ0star_ηy - PDMats.quad(Λ1star_ηy_h_old, β1star_ηy_h_old)) # posterior IG scale

            ## Important quantities for candidate
            D_h_cand = construct_Dh(h, X_h, μ_x_h_cand)

            Λ1star_ηy_h_cand = PDMat(D_h_cand'D_h_cand + model.state.Λ0star_ηy)
            β1star_ηy_h_cand = Λ1star_ηy_h_cand \ (Λβ0star_ηy + D_h_cand'y_h)

            a1_δy_cand = (model.state.ν_δy + n_h) / 2.0 # posterior IG shape
            b1_δy_cand = 0.5 * (model.state.ν_δy * model.state.s0_δy +
                                y_h'y_h + βΛβ0star_ηy - PDMats.quad(Λ1star_ηy_h_cand, β1star_ηy_h_cand) ) # posterior IG scale

            ## Compute acceptance ratio
            lar = lcc_ηlδx(h, indx_h, lNX_mat_cand, lωNX_vec_cand,
                model.state, μ_x_h_cand[1], lδ_x_h_cand[1],
                Λ1star_ηy_h_cand, a1_δy_cand, b1_δy_cand) -
                lcc_ηlδx(h, indx_h, model.state.lNX, model.state.lωNX_vec,
                    model.state, μ_x_h_old[1], lδ_x_h_old[1],
                    Λ1star_ηy_h_old, a1_δy_old, b1_δy_old)

            ## Decision and update
            if log(rand()) < lar # accept

                model.state.μ_x[h,:] = μ_x_h_cand
                model.state.δ_x[h,:] = δ_x_h_cand

                model.state.lNX[:,h] = lNX_mat_cand[:,h]
                model.state.lωNX_vec = copy(lωNX_vec_cand)

                ηlδ_x_out = ηlδ_x_cand
                model.state.accpt[h] += 1

                ## Full conditional draw for η_y ( use prec. prameterization )

                model.state.δ_y[h] = rand(InverseGamma(a1_δy_cand, b1_δy_cand))
                βstar_ηy = β1star_ηy_h_cand + (Λ1star_ηy_h_cand.chol.U \ randn((model.K+1)) .* sqrt(model.state.δ_y[h]))
                model.state.μ_y[h] = βstar_ηy[model.indx_ηy[:μ]]
                model.state.β_y[h,:] = βstar_ηy[model.indx_ηy[:β]]

            else # reject

                ηlδ_x_out = ηlδ_x_old

                ## Full conditional draw for η_y

                model.state.δ_y[h] = rand(InverseGamma(a1_δy_old, b1_δy_old))
                βstar_ηy = β1star_ηy_h_old + (Λ1star_ηy_h_old.chol.U \ randn((model.K+1)) .* sqrt(model.state.δ_y[h]))
                model.state.μ_y[h] = βstar_ηy[model.indx_ηy[:μ]]
                model.state.β_y[h,:] = βstar_ηy[model.indx_ηy[:β]]

            end
    end

    if model.state.adapt
        model.state.adapt_iter += 1
        model.state.runningsum_ηlδx[h,:] += ηlδ_x_out
        runningmean = model.state.runningsum_ηlδx[h,:] / float(model.state.adapt_iter)
        runningdev = ( ηlδ_x_out - runningmean )
        model.state.runningSS_ηlδx[h,:,:] = runningdev * runningdev' # the mean is changing, but this approx. is fine.
    end

    return nothing
end



function construct_Dh(h::Int, Xh::Array{T, 2}, μ_x_h::Array{T, 1}) where T <: Real
    D0 = broadcast(-, μ_x_h, Xh')'
    return hcat( ones(size(Xh,1)), D0 )
end

function lG0_ηlδx(μ_x::Array{T, 1}, β_x::Array{Array{T, 1}, 1}, lδ_x::Array{T, 1},
                  state::State_DPmRegJoint) where T <: Real

    K = length(μ_x)
    Q_μ = - 0.5 * PDMats.quad( state.Λ0_μx, (μ_x - state.μ0_μx) )
    Q_β = 0.0
    Q_δ = 0.0
    for k = 1:(K-1)
        Q_β -= 0.5 * PDMats.quad( state.Λ0_βx[k], (β_x[k] - state.β0_βx[k]) )
        # Q_δ -= ( (0.5*state.ν_δx[k] + 1.0)*lδ_x[k] +
        #           0.5*state.ν_δx[k]*state.s0_δx[k]/exp(lδ_x[k]) ) # build Jacobian into it
        Q_δ -= 0.5*( state.ν_δx[k]*lδ_x[k] + state.ν_δx[k]*state.s0_δx[k]/exp(lδ_x[k]) ) # Jacobian built in

    end
    # Q_δ -= ( (0.5*state.ν_δx[K] + 1.0)*lδ_x[K] +
    #           0.5*state.ν_δx[K]*state.s0_δx[K]/exp(lδ_x[K]) )
    Q_δ -= 0.5*( state.ν_δx[K]*lδ_x[K] + state.ν_δx[K]*state.s0_δx[K]/exp(lδ_x[K]) ) # Jacobian built in

    # J = sum(lδ_x) # Jacobian for log(δ) transformation
    # Jacobian now built in

    Q = Q_μ + Q_β + Q_δ # + J
    return Q
end
function lG0_ηlδx(μ_x::T, lδ_x::T,
                  state::State_DPmRegJoint) where T <: Real
    ## K = 1 case
    Q_μ = - 0.5 * Matrix(state.Λ0_μx)[1] * (μ_x - state.μ0_μx[1])^2.0
    Q_δ = - 0.5 * ( state.ν_δx[1]*lδ_x + state.ν_δx[1]*state.s0_δx[1]/exp(lδ_x) ) # Jacobian built in

    Q = Q_μ + Q_δ
    return Q
end

### log-collapsed conditional
function lcc_ηlδx(h::Int, indx_h::Array{Int, 1}, lNX_mat::Array{T, 2}, lωNX_vec::Array{T, 1},
    state::State_DPmRegJoint, μ_x_h::Array{T, 1}, β_x_h::Array{Array{T, 1}, 1}, lδ_x_h::Array{T, 1},
    Λ1star_ηy_h::PDMat{T}, a1_δy::T, b1_δy::T) where T <: Real

    # Q1
    Q1 = sum( lNX_mat[indx_h, h] )

    # Q2
    Q2 = sum(lωNX_vec)

    # Q3 , G0 including Jacobian
    Q3 = lG0_ηlδx(μ_x_h, β_x_h, lδ_x_h, state)

    # Q4
    Q4 = logdet(Λ1star_ηy_h)

    # Q5
    Q5 = log(b1_δy)

    return Q1 - Q2 + Q3 - 0.5*Q4 - (a1_δy)*Q5

end
function lcc_ηlδx(h::Int, indx_h::Array{Int, 1}, lNX_mat::Array{T, 2}, lωNX_vec::Array{T, 1},
    state::State_DPmRegJoint, μ_x_h::T, lδ_x_h::T,
    Λ1star_ηy_h::PDMat{T}, a1_δy::T, b1_δy::T) where T <: Real

    ## K = 1 case

    # Q1
    Q1 = sum( lNX_mat[indx_h, h] )

    # Q2
    Q2 = sum(lωNX_vec)

    # Q3 , G0 including Jacobian
    Q3 = lG0_ηlδx(μ_x_h, lδ_x_h, state)

    # Q4
    Q4 = logdet(Λ1star_ηy_h)

    # Q5
    Q5 = log(b1_δy)

    return Q1 - Q2 + Q3 - 0.5*Q4 - (a1_δy)*Q5

end
