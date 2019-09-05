# using Distributions
# using BayesInference

export ldensweight_mat, getEy, qnorm_mix, getQuant, getlogdens_EY;

function ldensweight_mat(X_pred::Array{T,2}, sims::Union{Array{Dict{Symbol,Any},1}, Array{Any,1}},
    γδc::Union{Float64, Array{T, 1}, Nothing}=Inf) where T <: Real

    # fill(1.0e6, size(sims[1][:β_y])[2]) # default γδc under variance inflation method

    nsim = length(sims)
    H, K = size(sims[1][:β_y])
    npred, Kx = size(X_pred)

    K == Kx || throw(error("X_pred dimensions not aligned with simuations."))
    useγ = haskey(sims[1], :γ)
    if useγ
        γglobal = ( typeof(sims[1][:γ]) == BitArray{1} )
        γlocal = ( typeof(sims[1][:γ]) == BitArray{2} )
    end

    out = Array{T,3}(undef, nsim, npred, H)

    if K > 1
        for ii = 1:nsim

            if useγ

                if γglobal

                    if γδc == Inf  # subset method
                        γindx = findall(sims[ii][:γ])
                        nγ = length( γindx )

                        if nγ == 0
                            lNX = zeros(Float64, npred, H)
                        elseif nγ == 1
                            δγ_x = δ_x_modify_γ(sims[ii][:δ_x], sims[ii][:γ], γδc)
                            lNX = lNXmat(X_pred[:,γindx],
                                         sims[ii][:μ_x][:,γindx], δγ_x) # npred by H matrix
                        elseif nγ > 1 && !isnothing(sims[ii][:β_x]) # indicates Σx_type is :full
                            βγ_x, δγ_x = βδ_x_modify_γ(sims[ii][:β_x],
                                        sims[ii][:δ_x],sims[ii][:γ], γδc)

                            lNX = lNXmat(X_pred[:,γindx], sims[ii][:μ_x][:,γindx],
                                         βγ_x, δγ_x) # npred by H matrix
                        elseif nγ > 1 && isnothing(sims[ii][:β_x]) # indicates Σx_type is :diag
                            lNX = lNXmat_Σdiag(X_pred[:,γindx], sims[ii][:μ_x][:,γindx], sims[ii][:δ_x][:,γindx]) # npred by H matrix
                        end

                    elseif isnothing(γδc) # integration method
                        γindx = findall(sims[ii][:γ])
                        nγ = length( γindx )

                        if nγ == 0
                            lNX = zeros(Float64, npred, H)
                        elseif nγ == 1 && !isnothing(sims[ii][:β_x]) # indicates Σx_type is :full
                            σ2xs = [ sqfChol_to_Σ( [ sims[ii][:β_x][k][h,:] for k = 1:(K-1) ], sims[ii][:δ_x][h,:] ).mat[γindx, γindx][1] for h = 1:H ]
                            lNX = lNXmat(X_pred[:,γindx], sims[ii][:μ_x][:,γindx], σ2xs) # npred by H matrix
                        elseif nγ == 1 && isnothing(sims[ii][:β_x]) # indicates Σx_type is :diag
                            σ2xs = deepcopy( sims[ii][:δ_x][:,γindx] )
                            lNX = lNXmat(X_pred[:,γindx], sims[ii][:μ_x][:,γindx], σ2xs) # npred by H matrix
                        elseif nγ > 1 && !isnothing(sims[ii][:β_x]) # indicates Σx_type is :full
                            Σxs = [ PDMat( sqfChol_to_Σ( [ sims[ii][:β_x][k][h,:] for k = 1:(K-1) ], sims[ii][:δ_x][h,:] ).mat[γindx, γindx] ) for h = 1:H ]
                            lNX = hcat( [ logpdf(MultivariateNormal(sims[ii][:μ_x][h,γindx], Σxs[h]), Matrix(X_pred[:,γindx]')) for h = 1:H ]... ) # n by H matrix
                        elseif nγ > 1 && isnothing(sims[ii][:β_x]) # indicates Σx_type is :diag
                            lNX = lNXmat_Σdiag(X_pred[:,γindx], sims[ii][:μ_x][:,γindx], sims[ii][:δ_x][:,γindx]) # npred by H matrix
                        end

                    else  # variance-inflation method
                        !isnothing(sims[ii][:β_x]) || throw("Variance-inflation variable selection not implemented for diagonal Σx.")
                        βγ_x, δγ_x = βδ_x_modify_γ(sims[ii][:β_x],
                                    sims[ii][:δ_x], sims[ii][:γ], γδc)

                        lNX = lNXmat(X_pred, sims[ii][:μ_x],
                                [ βγ_x[k] for k = 1:(K-1) ],
                                δγ_x) # npred by H
                    end

                elseif γlocal

                    if γδc == Inf  # subset method

                        lNX = zeros(Float64, npred, H)

                        for h = 1:H

                            γ_h = sims[ii][:γ][h,:]
                            γindx = findall(γ_h)
                            nγ = length( γindx )

                            if nγ == 0
                                # do nothing because column h is already zeros
                            elseif nγ == 1

                                lNX[:,h] = logpdf.(Normal(sims[ii][:μ_x][h,γindx[1]], sqrt(sims[ii][:δ_x][h,γindx[1]])), vec(X_pred[:,γindx]))
                            
                            elseif nγ > 1 && !isnothing(sims[ii][:β_x]) # indicates Σx_type is :full

                                βγ_x_h, δγ_x_h = βδ_x_h_modify_γ( [ sims[ii][:β_x][j][h,:] for j = 1:length(sims[ii][:β_x]) ], 
                                    sims[ii][:δ_x][h,:], γ_h, γδc) # either variance-inflated or subset
                
                                lNX[:,h] = lNX_sqfChol(Matrix(X_pred[:,γindx]'), sims[ii][:μ_x][h,γindx], βγ_x_h, δγ_x_h, true)
                
                            elseif nγ > 1 && isnothing(sims[ii][:β_x]) # indicates Σx_type is :diag

                                lNX[:,h] = lNX_Σdiag( Matrix(X_pred[:,γindx]'), sims[ii][:μ_x][h,γindx], sims[ii][:δ_x][h,γindx] )

                            end

                        end

                    elseif isnothing(γδc) # integration method

                        lNX = zeros(Float64, npred, H)

                        for h = 1:H

                            γ_h = sims[ii][:γ][h,:]
                            γindx = findall(γ_h)
                            nγ = length( γindx )

                            if nγ == 0
                               # do nothing because column h is already zeros
                            elseif nγ == 1 && !isnothing(sims[ii][:β_x]) # indicates Σx_type is :full

                                σ2x = sqfChol_to_Σ( [ sims[ii][:β_x][j][h,:] for j = 1:length(sims[ii][:β_x]) ], sims[ii][:δ_x][h,:] ).mat[γindx, γindx][1]
                                lNX[:,h] = logpdf.(Normal(sims[ii][:μ_x][h,γindx], sqrt(σ2x)), vec(X_pred[:,γindx]))

                            elseif nγ == 1 && isnothing(sims[ii][:β_x]) # indicates Σx_type is :diag

                                σ2x = deepcopy( sims[ii][:δ_x][h,γindx] )
                                lNX[:,h] = logpdf.(Normal(sims[ii][:μ_x][h,γindx], sqrt(σ2x)), vec(X_pred[:,γindx]))

                            elseif nγ > 1 && !isnothing(sims[ii][:β_x]) # indicates Σx_type is :full

                                Σx = PDMat( sqfChol_to_Σ( [ sims[ii][:β_x][j][h,:] for j = 1:length(sims[ii][:β_x]) ], sims[ii][:δ_x][h,:] ).mat[γindx, γindx] )                
                                lNX[:,h] = logpdf( MultivariateNormal(sims[ii][:μ_x][h,γindx], Σx), Matrix(X_pred[:,γindx]') )

                            elseif nγ > 1 && isnothing(sims[ii][:β_x]) # indicates Σx_type is :diag

                                lNX[:,h] = lNX_Σdiag( Matrix(X_pred[:,γindx]'), sims[ii][:μ_x][h,γindx], sims[ii][:δ_x][h,γindx] )

                            end

                        end

                    else  # variance-inflation method
                        throw("Variance-inflation variable selection not implemented for local variable selection.")
                    end

                end

            else
                β_x = deepcopy(sims[ii][:β_x])
                δ_x = deepcopy(sims[ii][:δ_x])

                if isnothing(β_x) # indicates Σx_type is :diag
                    lNX = lNXmat_Σdiag(X_pred, sims[ii][:μ_x], δ_x) # npred by H
                else # indicates Σx_type is :full
                    lNX = lNXmat(X_pred, sims[ii][:μ_x],
                        [ β_x[k] for k = 1:(K-1) ],
                        δ_x) # npred by H
                end
            end

            lωNX_mat = broadcast(+, permutedims(sims[ii][:lω]), lNX) # npred by H
            lωNX_vec = vec( BayesInference.logsumexp(lωNX_mat, 2) ) # npred vec

            out[ii,:,:] = broadcast(-, lωNX_mat, lωNX_vec) # normalized, npred by H
        end
    else  # K = 1
        for ii = 1:nsim

            if useγ

                if γglobal

                    if γδc == Inf || isnothing(γδc) # subset or integration method

                        if sims[ii][:γ][1]
                            lNX = lNXmat(vec(X_pred), vec(sims[ii][:μ_x]), vec(sims[ii][:δ_x])) # npred by H matrix
                        else
                            lNX = zeros(Float64, npred, H)
                        end

                    else  # variance-inflation method

                        δγ_x = δ_x_modify_γ(sims[ii][:δ_x], sims[ii][:γ], γδc)
                        lNX = lNXmat(vec(X_pred), vec(sims[ii][:μ_x]), vec(δγ_x))

                    end

                elseif γlocal

                    if γδc == Inf || isnothing(γδc) # subset or integration method

                        lNX = zeros(Float64, npred, H)

                        for h = 1:H

                            if sims[ii][:γ][h,1]
                                σ2x = deepcopy(sims[ii][:δ_x][h,1])
                                lNX[:,h] = logpdf.( Normal(sims[ii][:μ_x][h,1], sqrt(σ2x)), vec(X_pred) )
                            else
                                # do nothing, the column is already zeros
                            end

                        end

                    else  # variance-inflation method
                        throw("Variance-inflation variable selection not implemented for local variable selection.")
                    end

                end

            else
                lNX = lNXmat(vec(X_pred), vec(sims[ii][:μ_x]), vec(sims[ii][:δ_x])) # npred by H
            end

            lωNX_mat = broadcast(+, permutedims(sims[ii][:lω]), lNX) # npred by H
            lωNX_vec = vec( BayesInference.logsumexp(lωNX_mat, 2) ) # npred vec

            out[ii,:,:] = broadcast(-, lωNX_mat, lωNX_vec) # normalized, npred by H
        end
    end

    return out
end

# npred = 100
# X_pred = hcat(collect(range(Float32(-2.0), length=npred, stop=Float32(2.0))))
#
# ldw = ldensweight_mat(X_pred, sims)
#
# dw = exp.(ldw)
# sum(dw[1,1,:])
#
# mean_dw = mean(dw, dims=1)
# h = 6
# plot([scatter(x=X_pred[:,1], y=mean_dw[1,:,h], mode="scatter")])

function getEy(X_pred::Array{T,2}, dw_mat::Array{T,3}, sims::Union{Array{Dict{Symbol,Any},1}, Array{Any,1}}) where T <: Real
    nsim, npred, H = size(dw_mat)
    nsim2 = length(sims)
    H2, K = size(sims[1][:β_y])
    npred3, K3 = size(X_pred)

    nsim == nsim2 || throw(error("Dimension mismatch."))
    H == H2 || throw(error("Dimension mismatch."))
    npred == npred3 || throw(error("Dimension mismatch."))
    K == K3 || throw(error("Dimension mismatch."))

    useγ = haskey(sims[1], :γ)
    if useγ
        γglobal = ( typeof(sims[1][:γ]) == BitArray{1} )
        γlocal = ( typeof(sims[1][:γ]) == BitArray{2} )
    end

    out = zeros(T, nsim, npred)

    for ii = 1:nsim
        if useγ
            if γglobal
                βγ_y = β_y_modify_γ(sims[ii][:β_y], sims[ii][:γ])
            elseif γlocal
                βγ_y = sims[ii][:β_y] .* sims[ii][:γ]
            end
        else
            βγ_y = deepcopy(sims[ii][:β_y])
        end
        for j = 1:npred
            for h = 1:H
                Ey_h = sims[ii][:μ_y][h] - sum( βγ_y[h,:] .* (X_pred[j,:] - sims[ii][:μ_x][h,:]) )
                out[ii,j] += dw_mat[ii, j, h] * Ey_h
            end
        end
    end

    out
end

function qnorm_mix(q::Float64, μ::Vector{Float64}, σ::Vector{Float64}, w::Vector{Float64})
    all(w .>= 0.0) || throw(error("Weights must be non-negative"))
    q > 0.0 && q < 1.0 || throw(error("Quantile must be between 0 and 1."))

    sum(w) ≈ 1.0 ? nothing : w = w ./ sum(w)

    d = Distributions.Normal()

    if q >= 0.5
        z_max = Distributions.quantile(d, q)
        z_min = -z_max

        x_max = maximum( σ .* z_max .+ μ )
        x_min = minimum( σ .* z_min .+ μ )
    else
        z_min = Distributions.quantile(d, q)
        z_max = -z_min

        x_max = maximum( σ .* z_max .+ μ )
        x_min = minimum( σ .* z_min .+ μ )
    end

    if abs(q - w'cdf.(d, ( x_max .- μ ) ./ σ)) < 1.0e-4
        out = x_max
    elseif abs(q - w'cdf.(d, ( x_min .- μ ) ./ σ)) < 1.0e-4
        out = x_min
    elseif x_max > x_min
        f(x) = ( q - w'cdf.(d, ( x .- μ ) ./ σ) )
        out = Roots.find_zero(f, (x_min, x_max))
    end

    return out
end


function getQuant(q::Float64, X_pred::Array{T,2}, dw_mat::Array{T,3}, sims::Union{Array{Dict{Symbol,Any},1}, Array{Any,1}}) where T <: Real
    nsim, npred, H = size(dw_mat)
    nsim2 = length(sims)
    H2, K = size(sims[1][:β_y])
    npred3, K3 = size(X_pred)

    q > 0.0 && q < 1.0 || throw(error("Quantile must be between 0 and 1."))
    nsim == nsim2 || throw(error("Dimension mismatch."))
    H == H2 || throw(error("Dimension mismatch."))
    npred == npred3 || throw(error("Dimension mismatch."))
    K == K3 || throw(error("Dimension mismatch."))

    useγ = haskey(sims[1], :γ)
    if useγ
        γglobal = ( typeof(sims[1][:γ]) == BitArray{1} )
        γlocal = ( typeof(sims[1][:γ]) == BitArray{2} )
    end

    out = zeros(T, nsim, npred)

    for ii = 1:nsim
        if useγ
            if γglobal
                βγ_y = β_y_modify_γ(sims[ii][:β_y], sims[ii][:γ])
            elseif γlocal
                βγ_y = sims[ii][:β_y] .* sims[ii][:γ]
            end
        else
            βγ_y = deepcopy(sims[ii][:β_y])
        end

        σ = sqrt.(sims[ii][:δ_y])

        for j = 1:npred

            μ = [ sims[ii][:μ_y][h] - sum( βγ_y[h,:] .* (X_pred[j,:] - sims[ii][:μ_x][h,:]) ) for h = 1:H ]
            w = deepcopy(dw_mat[ii, j, :])

            try
                out[ii,j] = qnorm_mix(q, μ, σ, w)
            catch
                println("No root found, simid $(ii) Xpred indx $(j)")
                println("μ=$(μ)")
                println("σ=$(σ)")
                println("w=$(w)")
                qnorm_mix(q, μ, σ, w) # for debugging
            end
        end
    end

    out
end

# Ey = getEy(X_pred, dw, sims)

# ii = 2
# plot([scatter(x=X_pred[:,1], y=Ey[ii,:], mode="scatter")])

# mean_Ey = mean(Ey, dims=1)
# p10_Ey = [quantile(Ey[:,j], 0.1) for j = 1:size(Ey,2)]
# p50_Ey = [quantile(Ey[:,j], 0.5) for j = 1:size(Ey,2)]
# p90_Ey = [quantile(Ey[:,j], 0.9) for j = 1:size(Ey,2)]
#
# trace1 = [scatter(x=X_pred[:,1], y=mean_Ey[1,:], mode="scatter")]
# trace2 = [scatter(x=X[:,1], y=y, mode="markers")]
# data = [trace1, trace2]
# Plotly.plot(trace2)
#
# trace0 = [scatter(x=X_pred[:,1], y=p10_Ey, mode="scatter")]
# Plotly.plot(trace0)
#
# npred = 5
# X_pred = hcat(collect(range(Float32(-2.0), length=npred, stop=Float32(2.0))))
# ngrid = 100
# y_grid = collect(range(Float32(-2.0), length=ngrid, stop=Float32(2.0)))


function getlogdens_EY(X_pred::Array{T,2}, y_grid::Array{T,1},
    ldw_mat::Array{T,3}, sims::Union{Array{Dict{Symbol,Any},1}, Array{Any,1}}) where T <: Real

    ngrid = length(y_grid)
    nsim, npred, H = size(ldw_mat)
    nsim2 = length(sims)
    H2, K = size(sims[1][:β_y])
    npred3, K3 = size(X_pred)

    nsim == nsim2 || throw(error("Dimension mismatch."))
    H == H2 || throw(error("Dimension mismatch."))
    npred == npred3 || throw(error("Dimension mismatch."))
    K == K3 || throw(error("Dimension mismatch."))

    useγ = haskey(sims[1], :γ)
    if useγ
        γglobal = ( typeof(sims[1][:γ]) == BitArray{1} )
        γlocal = ( typeof(sims[1][:γ]) == BitArray{2} )
    end

    ldens0 = Array{T,4}(undef, (nsim, npred, ngrid, H))
    Ey = zeros(T, nsim, npred)

    for ii = 1:nsim
        if useγ
            if γglobal
                βγ_y = β_y_modify_γ(sims[ii][:β_y], sims[ii][:γ])
            elseif γlocal
                βγ_y = sims[ii][:β_y] .* sims[ii][:γ]
            end
        else
            βγ_y = deepcopy(sims[ii][:β_y])
        end
        for j = 1:npred
            for h = 1:H
                Ey_h = sims[ii][:μ_y][h] - sum( βγ_y[h,:] .* (X_pred[j,:] - sims[ii][:μ_x][h,:]) )
                Ey[ii,j] += exp(ldw_mat[ii, j, h]) * Ey_h
                for jj = 1:ngrid
                    ldens0[ii,j,jj,h] = ldw_mat[ii, j, h] + logpdf(Normal(Ey_h, sqrt(sims[ii][:δ_y][h])), y_grid[jj])
                end
            end
        end
    end

    ldens = reshape(BayesInference.logsumexp(ldens0,4), (nsim, npred, ngrid))

    return ldens, Ey
end

# ldw = ldensweight_mat(X_pred, sims)
# ldens, Ey = getlogdens_EY(X_pred, y_grid, ldw, sims)
# dens = exp.(ldens)
# mean_dens = mean(dens, dims=1)
#
# xind = 4
# println(X_pred[xind,:])
#
# trace0 = [scatter(x=y_grid, y=mean_dens[1,xind,:], mode="scatter")]
# Plotly.plot(trace0)
#
# plot(dens[:,xind,25])
# plot(dens[:,xind,75])
