# using Distributions
# using BayesInference

export ldensweight_mat, getEy, getlogdens_EY;

function ldensweight_mat(X_pred::Array{T,2}, sims::PostSims_DPmRegJoint) where T <: Real

    nsim, H, K = size(sims.β_y)
    npred, Kx = size(X_pred)

    K == Kx || throw(error("X_pred dimensions not aligned with simuations."))

    out = Array{T,3}(undef, nsim, npred, H)

    if K > 1
        for ii = 1:nsim

            lNX = lNXmat(X_pred, sims.μ_x[ii,:,:],
                [sims.β_x[k][ii,:,:] for k = 1:(K-1)],
                sims.δ_x[ii,:,:]) # npred by H

            lωNX_mat = broadcast(+, sims.lω[ii,:], lNX') # H by npred
            lωNX_vec = vec( BayesInference.logsumexp(lωNX_mat, 1) ) # npred vec

            out[ii,:,:] = broadcast(-, lωNX_mat', lωNX_vec) # normalized, npred by H
        end
    else
        for ii = 1:nsim

            lNX = lNXmat(vec(X_pred), vec(sims.μ_x[ii,:,:]), vec(sims.δ_x[ii,:,:])) # npred by H

            lωNX_mat = broadcast(+, sims.lω[ii,:], lNX') # H by npred
            lωNX_vec = vec( BayesInference.logsumexp(lωNX_mat, 1) ) # npred vec

            out[ii,:,:] = broadcast(-, lωNX_mat', lωNX_vec) # normalized, npred by H
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

function getEy(X_pred::Array{T,2}, dw_mat::Array{T,3}, sims::PostSims_DPmRegJoint) where T <: Real
    nsim, npred, H = size(dw_mat)
    nsim2, H2, K = size(sims.β_y)
    npred3, K3 = size(X_pred)

    nsim == nsim2 || throw(error("Dimension mismatch."))
    H == H2 || throw(error("Dimension mismatch."))
    npred == npred3 || throw(error("Dimension mismatch."))
    K == K3 || throw(error("Dimension mismatch."))

    out = zeros(T, nsim, npred)

    for ii = 1:nsim
        for j = 1:npred
            for h = 1:H
                Ey_h = sims.μ_y[ii,h] - sum( sims.β_y[ii,h,:] .* (X_pred[j,:] - sims.μ_x[ii,h,:]) )
                out[ii,j] += dw_mat[ii, j, h] * Ey_h
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
    ldw_mat::Array{T,3}, sims::PostSims_DPmRegJoint) where T <: Real

    ngrid = length(y_grid)
    nsim, npred, H = size(ldw_mat)
    nsim2, H2, K = size(sims.β_y)
    npred3, K3 = size(X_pred)

    nsim == nsim2 || throw(error("Dimension mismatch."))
    H == H2 || throw(error("Dimension mismatch."))
    npred == npred3 || throw(error("Dimension mismatch."))
    K == K3 || throw(error("Dimension mismatch."))

    ldens0 = Array{T,4}(undef, (nsim, npred, ngrid, H))
    Ey = zeros(T, nsim, npred)

    for ii = 1:nsim
        for j = 1:npred
            for h = 1:H
                Ey_h = sims.μ_y[ii,h] - sum( sims.β_y[ii,h,:] .* (X_pred[j,:] - sims.μ_x[ii,h,:]) )
                Ey[ii,j] += exp(ldw_mat[ii, j, h]) * Ey_h
                for jj = 1:ngrid
                    ldens0[ii,j,jj,h] = ldw_mat[ii, j, h] + logpdf(Normal(Ey_h, sqrt(sims.δ_y[ii,h])), y_grid[jj])
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
