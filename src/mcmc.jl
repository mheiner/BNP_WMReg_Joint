# mcmc.jl

export mcmc_DPmRegJoint!;

"""
mcmc_DPmRegJoint!(model, n_keep[, monitor, report_filename="out_progress.txt",
thin=1, report_freq=10000, samptypes])
"""
function mcmc_DPmRegJoint!(model::Mod_DPmRegJoint, n_keep::Int,
    monitor::Monitor_DPmRegJoint=Monitor_DPmRegJoint(true, false, false),
    report_filename::String="out_progress.txt", thin::Int=1,
    report_freq::Int=10000, samptypes=(Float32, Int32))

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(now()) for $(n_keep * thin) iterations.\n")

    sims = PostSims_DPmRegJoint(monitor, n_keep, n, K, H, samptypes)
    start_accpt = copy(model.accpt)
    start_iter = copy(model.iter)
    prev_accpt = copy(model.accpt) # set every report_freq iterations

    yX = hcat(y, X) # useful for allocation update

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            model.state.lΛ, model.state.lλ, model.state.Zζ = MetropIndep_ΛλZζ(model.S,
            model.state.lΛ, model.state.lλ, model.state.Zζ,
            model.prior.Λ, model.prior.λ, model.prior.Q,
            model.λ_indx,
            model.TT, model.R, model.M, model.K)

            Zandζnow = ZζtoZandζ(model.state.Zζ, model.λ_indx)


            model.iter += 1
            if model.iter % report_freq == 0
                write(report_file, "Iter $(model.iter) at $(now())\n")
                write(report_file, "Current Metropolis acceptance rates: $(float((model.accpt - prev_accpt) / report_freq))\n\n")
                prev_accpt = copy(model.accpt)
            end
        end

        if monitor.ηω

        end
        if monitor.S

        end
        if monitor.G0

        end
    end

    close(report_file)
    return (sims, float((model.accpt - start_accpt)/(model.iter - start_iter)) )
end
