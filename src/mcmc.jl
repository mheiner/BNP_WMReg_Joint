# mcmc.jl

"""
mcmc_DPmRegJoint!(model, n_keep[, save=true, report_filename="out_progress.txt",
thin=1, report_freq=10000])
"""
function mcmc_DPmRegJoint!(model::Mod_DPmRegJoint, n_keep::Int, save::Bool=true,
    report_filename::String="out_progress.txt", thin::Int=1,
    report_freq::Int=10000)

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(now()) for $(n_keep * thin) iterations.\n")

    if save
        sims = PostSimsMMTD( Matrix{Float64}(n_keep, model.M), # Λ
        [ Matrix{Float64}(n_keep, model.λ_indx.lens[m]) for m in 1:model.M ], # λ
        [ Matrix{Float64}(n_keep, model.K^(m+1)) for m in 1:model.M ], # Q
        Matrix{Int}(n_keep, monitor_len), # Zζ
        Matrix{Float64}(n_keep, model.M), # p1λ
        [ Matrix{Float64}(n_keep, model.K^m) for m in 1:model.M ] #= p1Q =# )
    end


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
            end
        end

        if save
            @inbounds sims.Λ[i,:] = exp.( model.state.lΛ )
            @inbounds sims.Zζ[i,:] = copy(model.state.Zζ[monitor_indx])
            for m in 1:model.M
                @inbounds sims.λ[m][i,:] = exp.( model.state.lλ[m] )
                @inbounds sims.Q[m][i,:] = exp.( vec( model.state.lQ[m] ) )
            end
        end
    end

    close(report_file)

    if save
        return sims
    else
        return model.iter
    end

end
