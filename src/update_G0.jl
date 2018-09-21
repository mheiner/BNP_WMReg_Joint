# update_G0.jl

function rpost_β0star_ηy(βstar_ηy::Array{T,2}, δ_y::Array{T,1}, Λ0star_ηy::PDMat{T},
                            β0star_ηy_mean::Array{T, 1}, β0star_ηy_Prec::PDMat{T})
                            where T <: Real

    ## Collect sufficient statistics
    nstar, Kstar = Size(βstar_ηy)
    sumδinv = 0.0
    sumyδinv = zeros(T, Kstar)
    for i = 1:nstar
        sumδinv += 1.0 / δ_y[i]
        sumyδinv += βstar_ηy[i,:] ./ δ_y[i]
    end

    ## Updated parameters
    ⁠Prec_out = PDMat(β0star_ηy_Prec + sumδinv .* Λ0star_ηy)
    a = β0star_ηy_Prec * β0star_ηy_mean + Λ0star_ηy * sumyδinv
    mean_out = Prec_out \ a

    ## Sample
    return mean_out + Prec_out.chol.U \ randn(Kstar)
end

function rpost_Λ0star_ηy(βstar_ηy::Array{T,2}, δ_y::Array{T,1}, β0star_ηy::Array{T,1},
                                 df::T, invSc::PDMat{T}) where T <: Real
    ## Collect sufficient statistic
    nstar, Kstar = Size(βstar_ηy)
    SS = zeros(T, Kstar, Kstar)
    for i = 1:nstar
        dev = (βstar_ηy[i,:] - β0star_ηy) ./ δ_y[i]
        SS +=  dev * dev'
    end

    ## Updated parameters
    df1 = df + nstar
    invSc1 = PDMat(invSc + SS)
    Sc1 = inv(invSc1) # is there some way around this?

    ## Sample
    return rand(Distributions.Wishart(df1, Sc1))
end

function rpost_IGs0_gammaPri(δ::Array{T,1}, ν::T, n0::T, s00::T) where T <: Real
    # δ is a vector of IG variates
    # ν is the degrees of freedom of the IG
    # the gamma prior has shape ν*n0/2 and rate ν*n0/(2*s00)
    cc = 0.5 * ν
    sh_out = cc * (n0 + lengh(δ))
    rate_out = cc * (n0 / s00 + sum( 1.0 ./ δ) )
    rand( Distributions.Gamma(sh_out, 1.0 / rate_out) )
end

function update_G0!(model::Model_DPmRegJoint)
    ii = sort(unique(model.state.S))

    βstar_ηy = hcat(μ_y[ii], β_y[ii,:])

    model.state.β0star_ηy = rpost_β0star_ηy( βstar_ηy,
            model.state.δ_y[ii], model.state.Λ0star_ηy,
            model.prior.β0star_ηy_mean, model.prior.β0star_ηy_Prec)

    model.state.Λ0star_ηy = rpost_Λ0star_ηy(βstar_ηy, model.state.δ_y[ii],
                                     model.state.β0star_ηy,
                                     model.prior.Λ0star_ηy_df,
                                     model.prior.Λ0star_ηy_df * model.prior.Λ0star_ηy_S0)

    model.state.s0_δy = rpost_IGs0_gammaPri(model.state.δ_y[ii], model.state.ν_δy,
                                model.prior.s0_δy_df, model.prior.s0_δy_s0)

    model.state.μ0_μx = BayesInference.rpost_MvN_knownPrec(model.state.μ_x[ii,:],
                                model.state.Λ0_μx,
                                model.prior.μ0_μx_mean, model.prior.μ0_μx_Prec)

    model.state.Λ0_μx = BayesInference.rpost_MvNprec_knownMean(model.state.μ_x[ii,:],
                                model.state.μ0_μx,
                                model.prior.Λ0_μx_df, model.prior.Λ0_μx_df * model.prior.Λ0_μx_S0)

    if model.K > 1
        for k = 1:(model.K - 1)
            model.state.β0_βx[k] = BayesInference.rpost_MvN_knownPrec(
                model.state.β_x[k][ii,:],
                model.state.Λ0_βx[k],
                model.prior.β0_βx_mean[k], model.prior.β0_βx_Prec[k])

            model.state.Λ0_βx[k] = BayesInference.rpost_MvNprec_knownMean(
                model.state.β_x[k][ii,:],
                model.state.β0_βx[k],
                model.prior.Λ0_βx_df[k], model.prior.Λ0_βx_df[k] * model.prior.Λ0_βx_S0[k])

            model.state.s0_δx[k] = rpost_IGs0_gammaPri(
                model.state.δ_x[ii,k], model.state.ν_δx[k],
                model.prior.s0_δx_df[k], model.prior.s0_δx_s0[k])
        end
    end

    model.state.s0_δx[model.K] = rpost_IGs0_gammaPri(model.state.δ_x[ii,model.K], model.state.ν_δx[model.K],
        model.prior.s0_δx_df[model.K], model.prior.s0_δx_s0[model.K])

    return nothing
end
