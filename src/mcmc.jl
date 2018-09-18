# mcmc.jl

export mcmc_DPmRegJoint!, adapt_DPmRegJoint!;

"""
mcmc_DPmRegJoint!(model, n_keep[, monitor, report_filename="out_progress.txt",
thin=1, report_freq=10000, samptypes])
"""
function mcmc_DPmRegJoint!(model::Model_DPmRegJoint, n_keep::Int,
    updatevars::Updatevars_DPmRegJoint=Updatevars_DPmRegJoint(true, true, true, true, true),
    monitor::Monitor_DPmRegJoint=Monitor_DPmRegJoint(true, false, false),
    report_filename::String="out_progress.txt", thin::Int=1,
    report_freq::Int=100, samptypes=(Float32, Int32))

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    sims = PostSims_DPmRegJoint(monitor, n_keep, model.n, model.K, model.H, samptypes)
    start_accpt = copy(model.state.accpt)
    start_iter = copy(model.state.iter)
    prev_accpt = copy(model.state.accpt) # set every report_freq iterations

    yX = hcat(model.y, model.X) # useful for allocation update
    model.state.lNX = lNXmat(model.X, model.state.μ_x, model.state.β_x, model.state.δ_x)
    model.state.lωNX_vec = lωNXvec(model.state.lω, model.state.lNX)

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            if updatevars.η
                Λβ0star_ηy = model.state.Λ0star_ηy*model.state.β0star_ηy
                βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)
                for h = 1:model.H
                    update_η_h_Met!(model, h, Λβ0star_ηy, βΛβ0star_ηy)
                end
            end

            if updatevars.S
                update_alloc!(model, yX)
            end

            if updatevars.lω
                update_vlω_mvSlice!(model)
            end

            if updatevars.α
                model.state.α = rand(BayesInference.post_alphaDP(model.H, model.state.lω[H],
                                        model.prior.α_sh, model.prior.α_rate))
            end

            if updatevars.G0

            end

            model.state.iter += 1
            if model.state.iter % report_freq == 0
                write(report_file, "Iter $(model.state.iter) at $(Dates.now())\n")
                write(report_file, "Current Metropolis acceptance rates: $(float((model.state.accpt - prev_accpt) / report_freq))\n\n")
                prev_accpt = copy(model.state.accpt)
            end
        end

        if monitor.ηlω
            sims.μ_y[i,:] = Array{samptypes[1]}(model.state.μ_y)  # nsim by H matrix
            sims.β_y[i,:,:] = Array{samptypes[1]}(model.state.β_y) # nsim by H by K array
            sims.δ_y[i,:] = Array{samptypes[1]}(model.state.δ_y)  # nsim by H matrix
            sims.μ_x[i,:,:] = Array{samptypes[1]}(model.state.μ_x)  # nsim by H by K array
            if model.K > 1
                for k = 1:(model.K-1)
                    sims.β_x[k][i,:,:] = Array{samptypes[1]}(model.state.β_x[k])   # vector of nsim by H by (k in (K-1):1) arrays
                end
            end
            sims.δ_x[i,:,:] = Array{samptypes[1]}(model.state.δ_x)   # nsim by H by K array
            sims.lω[i,:] = Array{samptypes[1]}(model.state.lω)   # nsim by H matrix
            sims.α[i] = float(samptypes[1])(model.state.α)    # nsim vector
        end

        if monitor.S
            sims.S[i,:] = Array{samptypes[2]}(model.state.S)     # nsim by n matrix
        end

        if monitor.G0
            sims.β0star_ηy[i,:] = Array{samptypes[1]}(model.state.β0star_ηy)    # nsim by K+1 matrix
            sims.Λ0star_ηy[i,:] = Array{samptypes[1]}(BayesInference.vech(Matrix(model.state.Λ0star_ηy), true))    # nsim by length(vech) matrix

            sims.ν_δy[i] = float(samptypes[1])(model.state.ν_δy)    # nsim vector; THIS IS ACTUALLY A FIXED QUANTITY
            sims.s0_δy[i] = float(samptypes[1])(model.state.s0_δy)    # nsim vector

            sims.μ0_μx[i,:] = Array{samptypes[1]}(model.state.μ0_μx)    # nsim by K matrix
            sims.Λ0_μx[i,:] = Array{samptypes[1]}(BayesInference.vech(Matrix(model.state.Λ0_μx), true))     # nsim by length(vech) matrix

            if model.K > 1
                for k = 1:(model.K-1)
                    sims.β0_βx[k][i,:] = Array{samptypes[1]}(model.state.β0_βx[k])  # vector of nsim by (k in (K-1):1) matrices
                    sims.Λ0_βx[k][i,:] = Array{samptypes[1]}(BayesInference.vech(Matrix(model.state.Λ0_βx[k]), true))  # vector of nsim by length(vech) matrices
                end
            end

            sims.ν_δx[i,:] = Array{samptypes[1]}(model.state.ν_δx)     # nsim by K matrix; THIS IS ACTUALLY A FIXED QUANTITY
            sims.s0_δx[i,:] = Array{samptypes[1]}(model.state.s0_δx)    # nsim by K matrix
        end
    end

    close(report_file)
    return (sims, float((model.state.accpt - start_accpt)/(model.state.iter - start_iter)) )
end


function adapt_DPmRegJoint!(model::Model_DPmRegJoint, n_iter_collectSS::Int, n_iter_scale::Int,
    updatevars::Updatevars_DPmRegJoint,
    report_filename::String="out_progress.txt",
    maxtries::Int=100, accpt_bnds::Vector{T}=[0.23, 0.40], adjust::Vector{T}=[0.77, 1.3]) where T <: Real

    collect_scale = 2.38^2 / float((model.K + model.K*(model.K+1)/2))

    ## initial runs
    write(report_file, "Beginning Adaptation Phase 1 of 3 (initial scaling) at $(Dates.now())\n")

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.H)

    while any(fails)
        tries += 1
        tries <= maxtries || throw(error("Exceeded maximum adaptation attempts."))

        sims, accpt = mcmc_DPmRegJoint!(model, n_iter_scale,
                            updatevars,
                            Monitor_DPmRegJoint(false, false, false),
                            report_filename, 1, 100)

        for h = 1:H
            fails[h] = (accpt[h] < accpt_bnds[1])
            if fails[h]
                model.state.cSig_ηlδx[h,:,:] = model.state.cSig_ηlδx[h,:,:] .* adjust[1]
            end
        end

    end

    ## cΣ collection
    write(report_file, "Beginning Adaptation Phase 2 of 3 (covariance collection) at $(Dates.now())\n")

    reset_adapt!(model)
    model.state.adapt = true

    sims, accpt = mcmc_DPmRegJoint!(model, n_iter_collect, updatevars,
                            Monitor_DPmRegJoint(false, false, false),
                                    report_filename, 1, 100)

    for h = 1:H
        model.state.cSig_ηlδx[h] = collect_scale .* model.state.runningSS_ηlδx[h,:,:] ./ float(model.sate.adapt_iter)
    end

    ## final scaling
    write(report_file, "Beginning Adaptation Phase 3 of 3 (final scaling) at $(Dates.now())\n")

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.H)

    while any(fails)
        tries += 1
        tries <= maxtries || throw(error("Exceeded maximum adaptation attempts."))

        sims, accpt = mcmc_DPmRegJoint!(model, n_iter_scale, updatevars,
                                Monitor_DPmRegJoint(false, false, false),
                                report_filename, 1, 100)

        for h = 1:H
            too_low = accpt_rate[h] < accpt_bnds[1]
            too_high = accpt_rate[h] > accpt_bnds[2]

            if too_low
                model.state.cSig_ηlδx[h,:,:] = model.state.cSig_ηlδx[h,:,:] .* adjust[1]
            elseif too_high
                model.state.cSig_ηlδx[h,:,:] = model.state.cSig_ηlδx[h,:,:] .* adjust[2]
            else
                fails[h] = false
            end
        end

    end

    write(report_file, "Beginning Adaptation Phase 3 of 3 (final scaling) at $(Dates.now())\n")
    reset_adapt!(model)
    model.state.adapt = true

    return nothing
end
