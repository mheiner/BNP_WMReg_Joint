# mcmc.jl

export mcmc_DPmRegJoint!, adapt_DPmRegJoint!;

"""
mcmc_DPmRegJoint!(model, n_keep[, monitor, report_filename="out_progress.txt",
thin=1, report_freq=10000, samptypes])
"""
function mcmc_DPmRegJoint!(model::Model_DPmRegJoint, n_keep::Int,
    updatevars::Updatevars_DPmRegJoint=Updatevars_DPmRegJoint(true, true, false, true, true, true),
    monitor::Monitor_DPmRegJoint=Monitor_DPmRegJoint(true, false, false, false),
    report_filename::String="out_progress.txt", thin::Int=1,
    report_freq::Int=100, samptypes=(Float32, Int32))

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    sims, symb_monitor = postSimsInit_DPmRegJoint(monitor, n_keep, model.state)
    start_accpt = deepcopy(model.state.accpt)
    start_iter = deepcopy(model.state.iter)
    prev_accpt = deepcopy(model.state.accpt) # set every report_freq iterations

    yX = hcat(model.y, model.X) # useful for allocation update

    if model.state.γδc == Inf # subset method for variable selection
        γindx_start = findall(model.state.γ)
        nγ_start = length(γindx_start)

        if nγ_start == 0
            model.state.lNX = zeros(Float64, model.n, model.H)
            model.state.lωNX_vec = zeros(Float64, model.n)
        elseif nγ_start == 1
            model.state.lNX = lNXmat(model.X[:,γindx_start],
                    model.state.μ_x[:,γindx_start], model.state.δ_x[:,γindx_start])
            model.state.lωNX_vec = lωNXvec(model.state.lω, model.state.lNX)
        elseif nγ_start > 1
            βγ_x_start, δγ_x_start = βδ_x_modify_γ(model.state.β_x, model.state.δ_x,
                                               model.state.γ, model.state.γδc)
            model.state.lNX = lNXmat(model.X[:,γindx_start],
                model.state.μ_x[:,γindx_start], βγ_x_start, δγ_x_start) # n by H matrix
            model.state.lωNX_vec = lωNXvec(model.state.lω, model.state.lNX)
        end
    else # variance-inflation method for variable selection
        if model.K > 1
            model.state.lNX = lNXmat(model.X, model.state.μ_x, model.state.β_x, model.state.δ_x)
        else
            model.state.lNX = lNXmat(vec(model.X), vec(model.state.μ_x), vec(model.state.δ_x))
        end
        model.state.lωNX_vec = lωNXvec(model.state.lω, model.state.lNX)
    end


    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            if updatevars.η
                Λβ0star_ηy = model.state.Λ0star_ηy * model.state.β0star_ηy
                βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)

                if model.K > 1
                    for h = 1:model.H
                        update_η_h_Met!(model, h, Λβ0star_ηy, βΛβ0star_ηy)
                    end
                else
                    for h = 1:model.H
                        update_η_h_Met_K1!(model, h, Λβ0star_ηy, βΛβ0star_ηy)
                    end
                end
            end

            if updatevars.γ
                update_γ!(model)
            end

            if updatevars.lω
                update_vlω_mvSlice!(model)
            end

            if updatevars.α
                model.state.α = rand(BayesInference.post_alphaDP(model.H, model.state.lω[model.H],
                    model.prior.α_sh, model.prior.α_rate))
            end

            if updatevars.G0
                update_G0!(model)
            end

            ## do last, followed by llik calculation
            if updatevars.S
                llik_num_mat = update_alloc!(model, yX)
            else
                llik_num_mat = llik_numerator(yX, model.K, model.H,
                        model.state.μ_y, model.state.β_y, model.state.δ_y,
                        model.state.μ_x, model.state.β_x, model.state.δ_x,
                        model.state.γ, model.state.γδc, model.state.lω)
            end
            model.state.llik = llik_DPmRegJoint(llik_num_mat, model.state.lωNX_vec)


            model.state.iter += 1
            if model.state.iter % report_freq == 0
                write(report_file, "Iter $(model.state.iter) at $(Dates.now())\n")
                write(report_file, "Log-likelihood $(model.state.llik)\n")
                write(report_file, "Current Metropolis acceptance rates: $(float((model.state.accpt - prev_accpt) / report_freq))\n\n")
                prev_accpt = deepcopy(model.state.accpt)
            end
        end

        sims[i] = BayesInference.deepcopyFields(model.state, symb_monitor)

    end

    close(report_file)
    return (sims, float((model.state.accpt - start_accpt)/(model.state.iter - start_iter)) )
end


function adjust_from_accptr(accptr::T, target::T, adjust_bnds::Array{T,1}) where T <: Real
    if accptr < target
        out = (1.0 - adjust_bnds[1]) * accptr / target + adjust_bnds[1]
    else
        out = (adjust_bnds[2] - 1.0) * (accptr - target) / (1.0 - target) + 1.0
    end

    return out
end


function adapt_DPmRegJoint!(model::Model_DPmRegJoint, n_iter_collectSS::Int, n_iter_scale::Int,
    updatevars::Updatevars_DPmRegJoint,
    report_filename::String="out_progress.txt",
    maxtries::Int=50, accptr_bnds::Vector{T}=[0.23, 0.44], adjust_bnds::Vector{T}=[0.01, 10.0]) where T <: Real

    target = StatsBase.mean(accptr_bnds)
    d = Int((model.K + model.K*(model.K+1)/2))
    collect_scale = 2.38^2 / float(d)

    ## initial runs
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 1 of 4 (initial scaling) at $(Dates.now())\n\n")
    close(report_file)

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.H)

    while any(fails)
        tries += 1
        if tries > maxtries
            println("Exceeded maximum adaptation attempts, Phase 1\n")
            report_file = open(report_filename, "a+")
            write(report_file, "\n\nExceeded maximum adaptation attempts, Phase 1\n\n")
            close(report_file)
            break
        end

        sims, accptr = mcmc_DPmRegJoint!(model, n_iter_scale,
        updatevars,
        Monitor_DPmRegJoint(false, false, false, false),
        report_filename, 1, 100)

        for h = 1:model.H
            fails[h] = (accptr[h] < accptr_bnds[1])
            if fails[h]
                model.state.cSig_ηlδx[h] *= adjust_from_accptr(accptr[h], target, adjust_bnds)
            end
        end

    end


    ## local scaling
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 2 of 4 (local scaling) at $(Dates.now())\n\n")
    close(report_file)

    model.state.adapt = false
    reset_adapt!(model)

    localtarget = target * 1.0

    for ii = 1:3

        for group in keys(model.indx_ηx)

            ig = model.indx_ηx[group]

            tries = 0
            fails = trues(model.H)
            while any(fails)
                tries += 1
                if tries > maxtries
                    println("Exceeded maximum adaptation attempts, Phase 2\n")
                    report_file = open(report_filename, "a+")
                    write(report_file, "\n\nExceeded maximum adaptation attempts, Phase 2\n\n")
                    close(report_file)
                    break
                end

                sims, accptr = mcmc_DPmRegJoint!(model, n_iter_scale, updatevars,
                Monitor_DPmRegJoint(false, false, false, false),
                report_filename, 1, 100)

                for h = 1:model.H
                    too_low = accptr[h] < (accptr_bnds[1] * 0.5)
                    too_high = accptr[h] > (accptr_bnds[2])

                    if too_low || too_high

                        fails[h] = true

                        tmp = Matrix(model.state.cSig_ηlδx[h])
                        σ = sqrt.(LinearAlgebra.diag(tmp))
                        ρ = StatsBase.cov2cor(tmp, σ)

                        σ[ig] *= adjust_from_accptr(accptr[h], localtarget, adjust_bnds)
                        tmp = StatsBase.cor2cov(ρ, σ)
                        tmp += Diagonal(fill(0.1*minimum(σ), size(tmp,1)))
                        model.state.cSig_ηlδx[h] = PDMat(tmp)

                    else
                        fails[h] = false
                    end

                end

            end

        end
    end


    ## cΣ collection
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 3 of 4 (covariance collection) at $(Dates.now())\n\n")
    close(report_file)

    reset_adapt!(model)
    model.state.adapt = true

    sims, accptr = mcmc_DPmRegJoint!(model, n_iter_collectSS, updatevars,
    Monitor_DPmRegJoint(false, false, false, false),
    report_filename, 1, 100)

    for h = 1:model.H
        Sighat = model.state.runningSS_ηlδx[h,:,:] / float(model.state.adapt_iter)
        minSighat = minimum(abs.(Sighat))
        SighatPD = Sighat + Matrix(Diagonal(fill(0.1*minSighat, d)))
        model.state.cSig_ηlδx[h] = PDMat(collect_scale * SighatPD)
    end

    ## final scaling
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 4 of 4 (final scaling) at $(Dates.now())\n\n")
    close(report_file)

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.H)

    while any(fails)
        tries += 1
        if tries > maxtries
            println("Exceeded maximum adaptation attempts, Phase 4\n")
            report_file = open(report_filename, "a+")
            write(report_file, "\n\nExceeded maximum adaptation attempts, Phase 4\n\n")
            close(report_file)
            break
        end

        sims, accptr = mcmc_DPmRegJoint!(model, n_iter_scale, updatevars,
        Monitor_DPmRegJoint(false, false, false, false),
        report_filename, 1, 100)

        for h = 1:model.H
            too_low = accptr[h] < accptr_bnds[1]
            too_high = accptr[h] > accptr_bnds[2]

            if too_low || too_high
                fails[h] = true
                model.state.cSig_ηlδx[h] *= adjust_from_accptr(accptr[h], target, adjust_bnds)
            else
                fails[h] = false
            end

        end

    end

    report_file = open(report_filename, "a+")
    write(report_file, "\n\nAdaptation concluded at $(Dates.now())\n\n")
    close(report_file)

    ## note that mcmc! also closes the report file
    reset_adapt!(model)
    model.state.adapt = false

    return nothing
end
