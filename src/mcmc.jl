# mcmc.jl

export mcmc!, adapt!; #, est_imp;

"""
mcmc!(model, n_keep[, monitor, report_filename="out_progress.txt",
thin=1, report_freq=10000])
"""
function mcmc!(model::Model_DPmRegJoint, n_keep::Int,
    updatevars::Updatevars_DPmRegJoint=Updatevars_DPmRegJoint(true, true, false, true, true, true),
    monitor::Monitor_DPmRegJoint=Monitor_DPmRegJoint(true, false, false, false),
    report_filename::Union{String, Nothing}="out_progress.txt", thin::Int=1,
    report_freq::Int=100)

    writeout = typeof(report_filename) == String

    ## output files
    if writeout
        report_file = open(report_filename, "a+")
        write(report_file, "Commencing MCMC at $(Dates.now()) from iteration $(model.state.iter) for $(n_keep * thin) iterations.\n\n")
        close(report_file)
    end

    sims, symb_monitor = postSimsInit_DPmRegJoint(monitor, n_keep, model.state)
    start_accpt = deepcopy(model.state.accpt)
    start_iter = deepcopy(model.state.iter)
    prev_accpt = deepcopy(model.state.accpt) # set every report_freq iterations

    yX = hcat(model.y, model.X) # useful for allocation update
    lfc_γon = zeros(model.K)

    ## Initialize lNX and lωNX_vec
    model.state.lNX, model.state.lωNX_vec = lNXmat_lωNXvec(model, model.state.γ)

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            if updatevars.γ
                lfc_γon = update_γ!(model)
            end

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
                if writeout
                    report_file = open(report_filename, "a+")
                    write(report_file, "Iter $(model.state.iter) at $(Dates.now())\n")
                    write(report_file, "Log-likelihood $(model.state.llik)\n")
                    write(report_file, "Current Metropolis acceptance rates: $(float((model.state.accpt - prev_accpt) / report_freq))\n")
                    write(report_file, "Current allocation: $(counts(model.state.S, 1:model.H))\n")
                    write(report_file, "Current final weight: $(exp(model.state.lω[model.H]))\n")
                    write(report_file, "Current variable selection: $(1*model.state.γ)\n\n")
                    close(report_file)
                end
                prev_accpt = deepcopy(model.state.accpt)
            end
        end

        sims[i] = BayesInference.deepcopyFields(model.state, symb_monitor)
        sims[i][:Scounts] = counts(model.state.S, 1:model.H)

        if updatevars.γ
            # sims[i][:lwimp] = model.state.lwimp
            sims[i][:lfc_on] = deepcopy(lfc_γon)
        end

    end

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


function adapt!(model::Model_DPmRegJoint;
    n_iter_collectSS::Int=1000, n_iter_scale::Int=500,
    adapt_thin::Int=1,
    updatevars::Updatevars_DPmRegJoint,
    report_filename::String="out_progress.txt",
    maxtries::Int=50,
    accptr_bnds::Vector{T}=[0.23, 0.44], adjust_bnds::Vector{T}=[0.01, 10.0]) where T <: Real

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

        sims, accptr = mcmc!(model, n_iter_scale,
            updatevars,
            Monitor_DPmRegJoint(false, false, false, false),
            tries == 1 ? report_filename : nothing,
            1, n_iter_scale)

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

                sims, accptr = mcmc!(model, n_iter_scale, updatevars,
                    Monitor_DPmRegJoint(false, false, false, false),
                    nothing,
                    1, n_iter_scale)

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
                        model.state.cSig_ηlδx[h] = PDMat_adj(tmp)

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
    model.state.adapt_thin = deepcopy(adapt_thin)

    sims, accptr = mcmc!(model, n_iter_collectSS, updatevars,
        Monitor_DPmRegJoint(false, false, false, false),
        report_filename, 1, n_iter_collectSS)

    for h = 1:model.H
        Sighat = model.state.runningSS_ηlδx[h,:,:] / float(model.state.adapt_iter)
        minSighat = minimum(abs.(Sighat))
        SighatPD = Sighat + Matrix(Diagonal(fill(0.1*minSighat, d)))
        model.state.cSig_ηlδx[h] = PDMat_adj(collect_scale * SighatPD)
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

        sims, accptr = mcmc!(model, n_iter_scale, updatevars,
            Monitor_DPmRegJoint(false, false, false, false),
            tries == 1 ? report_filename : nothing,
            1, n_iter_scale)

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

## estimate quantities with importance weights
# function est_imp(x::Vector{Float64}, weights::Vector{Float64})
#     sumw = sum(weights)
#     return sum(x .* weights) / sumw
# end
