# general.jl

export State_DPmRegJoint, Prior_DPmRegJoint, Model_DPmRegJoint,
    Monitor_DPmRegJoint, PostSims_DPmRegJoint, compute_lNX;

mutable struct State_DPmRegJoint

    ### Parameters
    # component (kernel parameters) η
    μ_y::Array{Float64, 1}  # H vector
    β_y::Array{Float64, 2}  # H by K matrix
    δ_y::Array{Float64, 1}  # H vector
    μ_x::Array{Float64, 2}  # H by K matrix
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing}  # vector of H by (k in (K-1):1) matrices
    δ_x::Array{Float64,2}   # H by K matrix

    # allocation states, weights
    S::Array{Int, 1}        # n vector
    lω::Array{Float64, 1}   # H vector
    v::Array{Float64, 1}    # H-1 vector
    α::Float64

    # G0
    β0_ηy::Array{Float64, 1}    # K+1 vector
    Λ0_ηy::PDMat{Float64}       # K+1 by K+1 precision matrix

    ν_δy::Float64
    s0_δy::Float64

    μ0_μx::Array{Float64, 1}    # K vector
    Λ0_μx::PDMat{Float64}       # K by K precision matrix

    β0_βx::Union{Array{Array{Float64, 1}, 1}, Nothing} # vector of (k in (K-1):1) vectors
    Λ0_βx::Union{Array{PDMat{Float64}, 1}, Nothing}    # vector of k by k (for k in (K-1):1) precison matrices

    ν_δx::Array{Float64, 1}     # K vector
    s0_δx::Array{Float64, 1}    # K vector

    ### other state objects
    iter::Int
    accpt::Array{Int, 1} # H vector
    cSig_ηlδx::Array{PDMat{Float64}, 1}
    adapt::Bool
    adapt_iter::Union{Int, Nothing}
    runningsum_ηx::Union{Array{Float64, 2}, Nothing} # H by (K + K(K+1)/2) matrix
    runningSS_ηx::Union{Array{Float64, 3}, Nothing}  # H by (K + K(K+1)/2) by (K + K(K+1)/2) matrix
    lNX::Array{Float64, 2} # n by H matrix of log(pdf(Normal(x_i))) under each obs i and allocation h
    lωNX_vec::Array{Float64, 1} # n vector with log( sum_j ωN(x_i) )

    # for coninuing an adapt phase
    State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
    S, lω, α, β0_ηy, Λ0_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx,
    adapt, adapt_iter, runningsum_ηx, runningSS_ηx, lNX, lωNX_vec) = new(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
        S, lω, lω_to_v(lω), α, β0_ηy, Λ0_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
        β0_βx, Λ0_βx, ν_δx, s0_δx,
        iter, accpt, cSig_ηlδx,
        adapt, adapt_iter, runningsum_ηx, runningSS_ηx, lNX, lωNX_vec)

    # for starting new
    State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
    S, lω, α, β0_ηy, Λ0_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    cSig_ηlδx, adapt) = new(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
        S, lω, lω_to_v(lω), α, β0_ηy, Λ0_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
        β0_βx, Λ0_βx, ν_δx, s0_δx,
        0, zeros(Int, length(lω)), cSig_ηlδx,
        adapt, 0,
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2),
                        Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros(Float64, 1, 1), zeros(Float64, 1) )

    # for starting new but not adapting
    State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
    S, lω, α, β0_ηy, Λ0_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx) = new(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
        S, lω, lω_to_v(lω), α, β0_ηy, Λ0_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
        β0_βx, Λ0_βx, ν_δx, s0_δx,
        iter, accpt, cSig_ηlδx,
        false, nothing, nothing, nothing,
        zeros(Float64, 1, 1), zeros(Float64, 1))
end

struct Prior_DPmRegJoint
    α_sh::Float64   # gamma shape
    α_rate::Float64 # gamma rate

    β0_ηy_mean::Array{Float64, 1}   # MVN mean vector
    β0_ηy_Cov::PDMat{Float64}       # MVN covariance matrix
    β0_ηy_Prec::PDMat{Float64}      # MVN precision matrix

    Λ0_ηy_df::Float64           # Wishart deg. of freedom
    Λ0_ηy_S0::PDMat{Float64}    # Prior harmonic mean of Λ0inv; inverse scale of Wishart divided by deg. of freedom

    s0_δy_df::Float64   # scaled inv. chi-square deg. of freedom
    s0_δy_s0::Float64   # scaled inv. chi-square harmonic mean

    μ0_μx_mean::Array{Float64, 1}   # MVN mean vector
    μ0_μx_Cov::PDMat{Float64}       # MVN covariance matrix
    μ0_μx_Prec::PDMat{Float64}      # MVN precision matrix

    Λ0_μx_df::Float64           # Wishart deg. of freedom
    Λ0_μx_S0::PDMat{Float64}    # Prior harmonic mean of Λ0inv; inverse scale of Wishart divided by deg. of freedom

    β0_βx_mean::Union{Array{Array{Float64, 1}, 1}, Nothing} # vector of MVN mean vectors
    β0_βx_Cov::Union{Array{PDMat{Float64}, 1}, Nothing}     # vector of MVN covariance matrices
    β0_βx_Prec::Union{Array{PDMat{Float64}, 1}, Nothing}    # vector of MVN precision matrices

    Λ0_βx_df::Union{Array{Float64, 1}, Nothing}         # vector of Wishart deg. of freedom
    Λ0_βx_S0::Union{Array{PDMat{Float64}, 1}, Nothing}  # vector of Prior harmonic mean of Λ0inv; inverse scale of Wishart divided by deg. of freedom

    s0_δx_df::Array{Float64, 1}   # vector of scaled inv. chi-square deg. of freedom
    s0_δx_s0::Array{Float64, 1}   # vector of scaled inv. chi-square harmonic mean

    Prior_DPmRegJoint(α_sh, α_rate, β0_ηy_mean, β0_ηy_Cov,
    Λ0_ηy_df, Λ0_ηy_S0, s0_δy_df, s0_δy_s0,
    μ0_μx_mean, μ0_μx_Cov, Λ0_μx_df, Λ0_μx_S0,
    β0_βx_mean, β0_βx_Cov, Λ0_βx_df, Λ0_βx_S0,
    s0_δx_df, s0_δx_s0) = new(α_sh, α_rate, β0_ηy_mean, β0_ηy_Cov, inv(β0_ηy_Cov),
    Λ0_ηy_df, Λ0_ηy_S0, s0_δy_df, s0_δy_s0,
    μ0_μx_mean, μ0_μx_Cov, inv(μ0_μx_Cov), Λ0_μx_df, Λ0_μx_S0,
    β0_βx_mean, β0_βx_Cov, [inv(β0_βx_Cov[k]) for k = 1:length(β0_βx_Cov)], Λ0_βx_df, Λ0_βx_S0,
    s0_δx_df, s0_δx_s0)
end

mutable struct Model_DPmRegJoint
    y::Array{Float64, 1}
    X::Array{Float64, 2}
    n::Int # length of data
    K::Int # number of predictors (columns in X)
    H::Int # DP truncation level
    prior::Prior_DPmRegJoint

    indx_ηy::Dict
    indx_ηx::Dict
    indx_β_x::Union{Dict, Nothing}

    state::State_DPmRegJoint # this is the only thing that should change

end
function Model_DPmRegJoint(y::Array{Float64, 1}, X::Array{Float64, 2},
    H::Int, prior::Prior_DPmRegJoint, state::State_DPmRegJoint)

    n = length(y)
    K = size(X, 2)

    indx_ηy = Dict( :μ => 1, :β => 2:(K+1), :δ => K+2 )

    if K > 1
        indx_ηx = Dict( :μ => 1:K, :β => (K+1):Int(K*(K+1)/2), :δ => Int(K*(K+1)/2 + 1):Int(K*(K+1)/2 + K) )
        indx_β_x = Dict()
        ii = 1
        jj = K - 1
        for k = 1:(K-1)
            indx_β_x[k] = Int(ii):Int(ii+jj-1)
            ii += jj
            jj -= 1
        end
    else
        indx_ηx = Dict( :μ => 1:K, :δ => Int(K+1):Int(2*K) )
        indx_β_x = nothing
    end

    return Model_DPmRegJoint(y, X, n, K, H, prior,
        indx_ηy, indx_ηx, indx_β_x, state)
end

mutable struct PostSims_DPmRegJoint
    μ_y::Array{<:Real, 2}  # nsim by H matrix
    β_y::Array{<:Real, 3}  # nsim by H by K array
    δ_y::Array{<:Real, 2}  # nsim by H matrix
    μ_x::Array{<:Real, 3}  # nsim by H by K array
    β_x::Array{Array{<:Real, 3}, 1}    # vector of nsim by H by (k in (K-1):1) arrays
    δ_x::Array{<:Real,3}   # nsim by H by K array

    # weights, alpha
    lω::Array{<:Real, 2}   # nsim by H matrix
    α::Array{<:Real, 1}    # nsim vector

    # states
    S::Array{<:Integer, 2}      # nsim by n matrix

    # G0
    β0_ηy::Array{<:Real, 2}    # nsim by K+1 matrix
    Λ0_ηy::Array{<:Real, 2}    # nsim by length(vech) matrix

    ν_δy::Array{<:Real, 1}     # nsim vector
    s0_δy::Array{<:Real, 1}    # nsim vector

    μ0_μx::Array{<:Real, 2}    # nsim by K matrix
    Λ0_μx::Array{<:Real, 2}    # nsim by length(vech) matrix

    β0_βx::Array{Array{<:Real, 2}, 1}  # vector of nsim by (k in (K-1):1) matrices
    Λ0_βx::Array{Array{<:Real, 2}, 1}  # vector of nsim by length(vech) matrices

    ν_δx::Array{<:Real, 2}     # nsim by K matrix
    s0_δx::Array{<:Real, 2}    # nsim by K matrix

PostSims_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
lω, α, S,
β0_ηy, Λ0_ηy, ν_δy, s0_δy,
μ0_μx, Λ0_μx, β0_βx, Λ0_βx, ν_δx, s0_δx) = new(μ_y, β_y, δ_y, μ_x, β_x, δ_x,
lω, α, S,
β0_ηy, Λ0_ηy, ν_δy, s0_δy,
μ0_μx, Λ0_μx, β0_βx, Λ0_βx, ν_δx, s0_δx)
end

mutable struct Monitor_DPmRegJoint
    ηlω::Bool
    S::Bool
    G0::Bool
end

PostSims_DPmRegJoint(m::Monitor_DPmRegJoint, n_keep::Int, n::Int, K::Int, H::Int, samptypes::Tuple{<:Real, <:Integer}) = PostSims_DPmRegJoint(
(m.ηlω ? Array{samptypes[1], 2}(undef, n_keep, H) : Array{samptypes[1], 2}(undef, 0, 0)), # μ_y
(m.ηlω ? Array{samptypes[1], 3}(undef, n_keep, H, K) : Array{samptypes[1], 3}(undef, 0, 0, 0)), # β_y
(m.ηlω ? Array{samptypes[1], 2}(undef, n_keep, H) : Array{samptypes[1], 2}(undef, 0, 0)), # δ_y
(m.ηlω ? Array{samptypes[1], 3}(undef, n_keep, H, K) : Array{samptypes[1], 3}(undef, 0, 0, 0)), # μ_x
(m.ηlω && K > 1 ? [ Array{samptypes[1], 3}(undef, n_keep, H, k) for k = (K-1):-1:1 ] : [ Array{samptypes[1], 3}(undef, 0, 0, 0) for k = 1:2 ] ), # β_x
(m.ηlω ? Array{samptypes[1], 3}(undef, n_keep, H, K) : Array{samptypes[1], 3}(undef, 0, 0, 0)), # δ_x
(m.ηlω ? Array{samptypes[1], 2}(undef, n_keep, H) : Array{samptypes[1], 2}(undef, 0, 0)), # lω
(m.ηlω ? Array{samptypes[1], 1}(undef, n_keep) : Array{samptypes[1], 1}(undef, 0)), # α
(m.S ? Array{samptypes[2], 2}(undef, n_keep, n) : Array{samptypes[2], 2}(undef, 0, 0)), # S
(m.G0 ? Array{samptypes[1], 2}(undef, n_keep, K+1) : Array{samptypes[1], 2}(undef, 0, 0)), # β0_ηy
(m.G0 ? Array{samptypes[1], 2}(undef, n_keep, (K+1)*(K+2)/2) : Array{samptypes[1], 2}(undef, 0, 0)), # Λ0_ηy
(m.G0 ? Array{samptypes[1], 1}(undef, n_keep) : Array{samptypes[1], 1}(undef, 0)), # ν_δy
(m.G0 ? Array{samptypes[1], 1}(undef, n_keep) : Array{samptypes[1], 1}(undef, 0)), # s0_δy
(m.G0 ? Array{samptypes[1], 2}(undef, n_keep, K) : Array{samptypes[1], 2}(undef, 0, 0)), # μ0_μx
(m.G0 ? Array{samptypes[1], 2}(undef, n_keep, (K)*(K+1)/2) : Array{samptypes[1], 2}(undef, 0, 0)), # Λ0_μx
(m.G0 && K > 1 ? [ Array{samptypes[1], 2}(undef, n_keep, k) for k = (K-1):-1:1 ] : [ Array{samptypes[1], 2}(undef, 0, 0) for k = 1:2 ] ), # β0_βx
(m.G0 && K > 1 ? [ Array{samptypes[1], 2}(undef, n_keep, k*(k+1)/2) for k = (K-1):-1:1 ] : [ Array{samptypes[1], 2}(undef, 0, 0) for k = 1:2 ] ), # Λ0_βx
(m.G0 ? Array{samptypes[1], 2}(undef, n_keep, K) : Array{samptypes[1], 2}(undef, 0, 0)), # ν_δx
(m.G0 ? Array{samptypes[1], 2}(undef, n_keep, K) : Array{samptypes[1], 2}(undef, 0, 0)) ) # s0_δx

function lNXmat(X::Array{T, 2}, μ::Array{T, 2}, β::Array{Array{T, 2}, 1}, δ::Array{T, 2}) where T <: Real
    hcat( [lNX_sqfChol( Matrix(X'), μ[h,:], [ β[k][h,:] for k = 1:(size(X,2)-1) ], δ[h,:] )
            for h = 1:size(μ, 1) ]...)
end

function lωNXvec(lω::Array{T, 1}, lNX_mat::Array{T, 1}) where T <: Real
    lωNX_mat = broadcast(+, lω, lNX_mat') # H by n
    BayesInference.logsumexp(lωNX_mat, 1) # n vector
end
