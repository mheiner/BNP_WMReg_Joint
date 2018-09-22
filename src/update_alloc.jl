# update_alloc.jl

function update_alloc!(model::Model_DPmRegJoint, yX::Array{T,2}) where T <: Real

    μ = hcat(model.state.μ_y, model.state.μ_x)
    β = ( model.K > 1 ? [model.state.β_y, model.state.β_x...] : [ model.state.β_y ])
    δ = hcat(model.state.δ_y, model.state.δ_x)

    # the rest could be done in parallel
    lW = hcat([ model.state.lω[h] .+
                lNX_sqfChol( Matrix(yX'), μ[h,:], [ β[k][h,:] for k = 1:model.K ], δ[h,:] )
                for h = 1:model.H
              ]...) # lW is a n by H matrix

              # lNXmat(model.X, model.state.μ_x, model.state.β_x, model.state.δ_x)

    ms = maximum(lW, dims=2) # maximum across columns
    bc_lWmimusms = broadcast(-, lW, ms)
    W = exp.(bc_lWmimusms)

    alloc_new = [ sample(StatsBase.Weights(W[i,:])) for i = 1:model.n ]
    model.state.S = alloc_new

    model.state.n_occup = length(unique(alloc_new))

    return nothing
end
