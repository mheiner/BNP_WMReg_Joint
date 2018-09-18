# mcmc.jl

export mcmc_DPmRegJoint!, adapt_DPmRegJoint!;

"""
mcmc_DPmRegJoint!(model, n_keep[, monitor, report_filename="out_progress.txt",
thin=1, report_freq=10000, samptypes])
"""
function mcmc_DPmRegJoint!(model::Model_DPmRegJoint, n_keep::Int,
    monitor::Monitor_DPmRegJoint=Monitor_DPmRegJoint(true, false, false),
    report_filename::String="out_progress.txt", thin::Int=1,
    report_freq::Int=100, samptypes=(Float32, Int32))

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()()) for $(n_keep * thin) iterations.\n")

    sims = PostSims_DPmRegJoint(monitor, n_keep, model.n, model.K, model.H, samptypes)
    start_accpt = copy(model.state.accpt)
    start_iter = copy(model.state.iter)
    prev_accpt = copy(model.state.accpt) # set every report_freq iterations

    yX = hcat(y, X) # useful for allocation update
    model.state.lNX = lNXmat(model.X, model.state.μ_x, model.state.β_x, model.state.δ_x)
    model.state.lωNX_vec = lωNXvec(model.state.lω, model.state.lNX)

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            Λβ0star_ηy = model.state.Λ0star_ηy*model.state.β0star_ηy
            βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)
            for h = 1:model.H
                update_η_h_Met!(model, h, Λβ0star_ηy, βΛβ0star_ηy)
            end
            update_alloc!(model, yX)
            update_vlω_mvSlice!(model)
            model.state.α = rand(BayesInference.post_alphaDP(model.H, model.state.lω[H],
                                    model.prior.α_sh, model.prior.α_rate))

            # update G0

            model.state.iter += 1
            if model.state.iter % report_freq == 0
                write(report_file, "Iter $(model.state.iter) at $(Dates.now()())\n")
                write(report_file, "Current Metropolis acceptance rates: $(float((model.state.accpt - prev_accpt) / report_freq))\n\n")
                prev_accpt = copy(model.state.accpt)
            end
        end

        if monitor.ηlω

        end
        if monitor.S

        end
        if monitor.G0

        end
    end

    close(report_file)
    return (sims, float((model.state.accpt - start_accpt)/(model.state.iter - start_iter)) )
end


function adapt_DPmRegJoint!(model::Model_DPmRegJoint, n_iter_collectSS::Int, n_iter_scale::Int,
    report_filename::String="out_progress.txt",
    maxtries::Int=100, accpt_bnds::Vector{T}=[0.23, 0.40], adjust::Vector{T}=[0.77, 1.3]) where T <: Real

    collect_scale = 2.38^2 / float((model.K + model.K*(model.K+1)/2))

    ## initial runs
    write(report_file, "Beginning Adaptation Phase 1 of 3 (initial scaling) at $(Dates.now()())\n")

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.H)

    while any(fails)
        tries += 1
        tries <= maxtries || throw(error("Exceeded maximum adaptation attempts."))

        sims, accpt = mcmc_DPmRegJoint!(model, n_iter_scale, Monitor_DPmRegJoint(false, false, false),
                                        report_filename, 1, 100)

        for h = 1:H
            fails[h] = (accpt[h] < accpt_bnds[1])
            if fails[h]
                model.state.cSig_ηlδx[h,:,:] = model.state.cSig_ηlδx[h,:,:] .* adjust[1]
            end
        end

    end

    ## cΣ collection
    write(report_file, "Beginning Adaptation Phase 2 of 3 (covariance collection) at $(Dates.now()())\n")

    reset_adapt!(model)
    model.state.adapt = true

    sims, accpt = mcmc_DPmRegJoint!(model, n_iter_collect, Monitor_DPmRegJoint(false, false, false),
                                    report_filename, 1, 100)

    for h = 1:H
        model.state.cSig_ηlδx[h] = collect_scale .* model.state.runningSS_ηlδx[h,:,:] ./ float(model.sate.adapt_iter)
    end

    ## final scaling
    write(report_file, "Beginning Adaptation Phase 3 of 3 (final scaling) at $(Dates.now()())\n")

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.H)

    while any(fails)
        tries += 1
        tries <= maxtries || throw(error("Exceeded maximum adaptation attempts."))

        sims, accpt = mcmc_DPmRegJoint!(model, n_iter_scale, Monitor_DPmRegJoint(false, false, false),
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

    write(report_file, "Beginning Adaptation Phase 3 of 3 (final scaling) at $(Dates.now()())\n")
    reset_adapt!(model)
    model.state.adapt = true

    return nothing
end
