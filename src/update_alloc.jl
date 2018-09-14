# update_alloc.jl

function update_alloc!(model::Mod_DPmRegJoint, yX::Array{T,2}) where T <: Real

    μ = hcat(model.state.μ_y, model.state.μ_x)
    β = [β_y, β_x...]
    δ = hcat(model.state.δ_y, model.state.δ_x)

    # this could be done in parallel
    lW = hcat([ model.state.lω[h] .+
                logpdf( MvNormal(μ[h,:],
                                 sqfChol2Σ([ B[k][h,:] for k = 1:K ],
                                           δ[h,:])),
                        Matrix(yX')
                      )
            for h = 1:model.H
         ]...) # lW is a n by H matrix

    ms = maximum(lW, dims=2) # maximum across columns
    bc_lWmimusms = broadcast(-, lW, ms)
    W = exp.(bc_lWmimusms)

    alloc_out = [ sample(Weights(W[i,:])) for i = 1:model.n ]

    model.state.S = alloc_out
end