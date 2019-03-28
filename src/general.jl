# general.jl

export State_DPmRegJoint, init_state_DPmRegJoint,
    Prior_DPmRegJoint, Model_DPmRegJoint,
    Monitor_DPmRegJoint, Updatevars_DPmRegJoint,
    PostSims_DPmRegJoint, lNXmat, reset_adapt!,
    llik_DPmRegJoint;

mutable struct State_DPmRegJoint

    ### Parameters
    # component (kernel parameters) η
    μ_y::Array{Float64, 1}  # H vector
    β_y::Array{Float64, 2}  # H by K matrix
    δ_y::Array{Float64, 1}  # H vector
    μ_x::Array{Float64, 2}  # H by K matrix
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing}  # vector of H by (k in (K-1):1) matrices
    δ_x::Array{Float64,2}   # H by K matrix

    # variable selection
    γ::Union{BitArray{1}, BitArray{2}} # K vector (for global, H by K matrix for local, which is not implemented yet)
    γδc::Union{Float64, Array{Float64, 1}, Nothing} # scale factors for δ_x
    π_γ::Array{Float64, 1} # K vector of probabilities that γ_k = 1 (true in code)

    # allocation states, weights
    S::Array{Int, 1}        # n vector
    lω::Array{Float64, 1}   # H vector
    v::Array{Float64, 1}    # H-1 vector
    α::Float64

    # G0
    β0star_ηy::Array{Float64, 1}    # K+1 vector
    Λ0star_ηy::PDMat{Float64}       # K+1 by K+1 precision matrix

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
    runningsum_ηlδx::Union{Array{Float64, 2}, Nothing} # H by (K + K(K+1)/2) matrix
    runningSS_ηlδx::Union{Array{Float64, 3}, Nothing}  # H by (K + K(K+1)/2) by (K + K(K+1)/2) matrix
    lNX::Array{Float64, 2} # n by H matrix of log(pdf(Normal(x_i))) under each obs i and allocation h
    lωNX_vec::Array{Float64, 1} # n vector with log( sum_j ωN(x_i) )
    n_occup::Int
    llik::Float64

end

### Outer constructors for State_DPmRegJoint
## for coninuing an adapt phase
function State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ,
    S, lω, α, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx,
    adapt, adapt_iter, runningsum_ηlδx, runningSS_ηlδx, lNX, lωNX_vec, llik)

    State_DPmRegJoint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ),
        deepcopy(S), deepcopy(lω), lω_to_v(lω), α, deepcopy(β0star_ηy),
        deepcopy(Λ0star_ηy), ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        iter, accpt, deepcopy(cSig_ηlδx),
        adapt, adapt_iter, deepcopy(runningsum_ηlδx), deepcopy(runningSS_ηlδx),
        deepcopy(lNX), deepcopy(lωNX_vec), length(unique(S)), llik)
end

## for starting new
function State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ,
    S, lω, α, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    cSig_ηlδx, adapt)

    State_DPmRegJoint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ),
        deepcopy(S), deepcopy(lω), lω_to_v(lω), α, deepcopy(β0star_ηy), deepcopy(Λ0star_ηy),
        ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        0, zeros(Int, length(lω)), deepcopy(cSig_ηlδx),
        adapt, 0,
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2),
                        Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros(Float64, 1, 1), zeros(Float64, 1),
        length(unique(S)), 0.0 )
end

## for starting new but not adapting
function State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ,
    S, lω, α, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx)

    State_DPmRegJoint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ),
        deepcopy(S), deepcopy(lω), lω_to_v(lω), α, deepcopy(β0star_ηy),
        deepcopy(Λ0star_ηy), ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        iter, accpt, deepcopy(cSig_ηlδx),
        false, nothing, nothing, nothing,
        zeros(Float64, 1, 1), zeros(Float64, 1),
        length(unique(S)), 0.0)
end

mutable struct Prior_DPmRegJoint
    α_sh::Float64   # gamma shape
    α_rate::Float64 # gamma rate

    π_sh::Array{Float64, 2} # Beta shape parameters

    β0star_ηy_mean::Array{Float64, 1}   # MVN mean vector
    β0star_ηy_Cov::PDMat{Float64}       # MVN covariance matrix
    β0star_ηy_Prec::PDMat{Float64}      # MVN precision matrix

    Λ0star_ηy_df::Float64           # Wishart deg. of freedom
    Λ0star_ηy_S0::PDMat{Float64}    # Prior harmonic mean of Λ0inv; inverse scale of Wishart divided by deg. of freedom

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
end

### Outer constructors for Prior_DPmRegJoint
## automatic creation of precision matrices from covariance matrices
function Prior_DPmRegJoint(α_sh, α_rate, π_sh, β0star_ηy_mean, β0star_ηy_Cov,
    Λ0star_ηy_df, Λ0star_ηy_S0, s0_δy_df, s0_δy_s0,
    μ0_μx_mean, μ0_μx_Cov, Λ0_μx_df, Λ0_μx_S0,
    β0_βx_mean, β0_βx_Cov, Λ0_βx_df, Λ0_βx_S0,
    s0_δx_df, s0_δx_s0)

    Prior_DPmRegJoint(α_sh, α_rate, deepcopy(π_sh), deepcopy(β0star_ηy_mean),
        deepcopy(β0star_ηy_Cov), inv(β0star_ηy_Cov),
        deepcopy(Λ0star_ηy_df), deepcopy(Λ0star_ηy_S0), s0_δy_df, s0_δy_s0,
        deepcopy(μ0_μx_mean), deepcopy(μ0_μx_Cov), inv(μ0_μx_Cov), Λ0_μx_df,
        deepcopy(Λ0_μx_S0), deepcopy(β0_βx_mean), deepcopy(β0_βx_Cov),
        ( typeof(β0_βx_Cov) == Nothing ? nothing : [inv(β0_βx_Cov[k]) for k = 1:length(β0_βx_Cov)] ),
        deepcopy(Λ0_βx_df), deepcopy(Λ0_βx_S0),
        deepcopy(s0_δx_df), deepcopy(s0_δx_s0))
end

## default prior spec
function Prior_DPmRegJoint(K::Int, H::Int)

    Prior_DPmRegJoint(3.0, # α_sh
    1.0, # α_rate
    fill(0.5, K, 2), # π_sh
    zeros(K+1), # β0star_ηy_mean
    PDMat(Matrix(Diagonal(vcat(9.0, fill(1.0, K))))), # β0star_ηy_Cov
    5.0*(K+1+2), # Λ0star_ηy_df
    PDMat(Matrix(Diagonal(fill(9.0, K+1)))), # Λ0star_ηy_S0
    5.0, # s0_δy_df
    0.1, # s0_δy_s0
    zeros(K), # μ0_μx_mean
    PDMat(Matrix(Diagonal(fill(9.0, K)))), # μ0_μx_Cov
    5.0*(K+2), # Λ0_μx_df
    PDMat(Matrix(Diagonal(fill(9.0, K)))), # Λ0_μx_S0
    (K > 1 ? [zeros(k) for k = (K-1):-1:1] : nothing), # β0_βx_mean
    (K > 1 ? [ PDMat(Matrix(Diagonal(fill(9.0, k)))) for k = (K-1):-1:1 ] : nothing), # β0_βx_Cov
    (K > 1 ? fill(5.0*(K+2), K-1) : nothing), # Λ0_βx_df
    (K > 1 ? [ PDMat(Matrix(Diagonal(fill(9.0, k)))) for k = (K-1):-1:1 ] : nothing), # Λ0_βx_S0
    fill(5.0, K), # s0_δx_df
    fill(1.0, K))

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

    Model_DPmRegJoint(y, X, n, K, H, prior,
        indx_ηy, indx_ηx, indx_β_x, state) = new(deepcopy(y), deepcopy(X), copy(n),
        deepcopy(K), deepcopy(H), deepcopy(prior),
        deepcopy(indx_ηy), deepcopy(indx_ηx), deepcopy(indx_β_x), deepcopy(state))
end

function Model_DPmRegJoint(y::Array{Float64, 1}, X::Array{Float64, 2},
    H::Int, prior::Prior_DPmRegJoint, state::State_DPmRegJoint)

    n = length(y)
    nx, K = size(X)

    n == nx || throw(error("X and y size mismatch."))

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

    return Model_DPmRegJoint(deepcopy(y), deepcopy(X), n, K, deepcopy(H), deepcopy(prior),
        indx_ηy, indx_ηx, indx_β_x, deepcopy(state))
end

mutable struct Monitor_DPmRegJoint
    ηlω::Bool
    γ::Bool
    S::Bool
    G0::Bool
end

mutable struct Updatevars_DPmRegJoint
    η::Bool
    lω::Bool
    γ::Bool
    α::Bool
    S::Bool
    G0::Bool
end

function postSimsInit_DPmRegJoint(m::Monitor_DPmRegJoint, n_keep::Int, init_state::State_DPmRegJoint)

    symb = [ :n_occup, :llik ]

    if m.ηlω
        push!(symb, [:μ_y, :β_y, :δ_y, :μ_x, :β_x, :δ_x, :lω, :α]...)
    end

    if m.γ
        push!(symb, [:γ, :π_γ]...)
    end

    if m.S
        push!(symb, :S)
    end

    if m.G0
        push!(symb, [:β0star_ηy, :Λ0star_ηy, :s0_δy,
                     :μ0_μx, :Λ0_μx, :β0_βx, :Λ0_βx, :s0_δx]...)
    end

    sims = fill( BayesInference.deepcopyFields(init_state, symb), n_keep )

    return sims, symb
end


function lNXmat(X::Array{T, 2}, μ::Array{T, 2}, β::Array{Array{T, 2}, 1}, δ::Array{T, 2}) where T <: Real
    hcat( [lNX_sqfChol( Matrix(X'), μ[h,:], [ β[k][h,:] for k = 1:(size(X,2)-1) ], δ[h,:] )
            for h = 1:size(μ, 1) ]...)
end
function lNXmat(x::Union{Array{T,1}, Array{T,2}},
                μ::Union{Array{T,1}, Array{T,2}},
                δ::Union{Array{T,1}, Array{T,2}}) where T <: Real
    # K = 1 case

    x = vec(x)
    μ = vec(μ)
    δ = vec(δ)
    hcat( [ logpdf.(Normal(μ[h], sqrt(δ[h])), x) for h = 1:length(μ) ]...)
end


function lωNXvec(lω::Array{T, 1}, lNX_mat::Array{T, 2}) where T <: Real
    lωNX_mat = broadcast(+, lω, lNX_mat') # H by n
    vec( BayesInference.logsumexp(lωNX_mat, 1) ) # n vector
end

function reset_adapt!(model::Model_DPmRegJoint)
    model.state.adapt_iter = 0
    ncov = Int(model.K + model.K*(model.K+1)/2)
    model.state.runningsum_ηlδx = zeros( Float64, model.H, ncov )
    model.state.runningSS_ηlδx = zeros( Float64, model.H, ncov, ncov )
    return nothing
end

function init_state_DPmRegJoint(n::Int, K::Int, H::Int,
    prior::Prior_DPmRegJoint, random::Bool=true, γglobal::Bool=true)

    if random
        s0_δx = [ rand(InverseGamma(prior.s0_δx_df[k]/2.0, prior.s0_δx_df[k]*prior.s0_δx_s0[k]/2.0)) for k = 1:K ]
        ν_δx = fill(5.0, K)
        Λ0_βx = ( K > 1 ? [ PDMat(rand(Wishart(prior.Λ0_βx_df[k], inv(prior.Λ0_βx_S0[k])/prior.Λ0_βx_df[k]))) for k = 1:(K-1) ] : nothing)
        β0_βx = ( K > 1 ? [ rand(MvNormal(prior.β0_βx_mean[k], prior.β0_βx_Cov[k])) for k = 1:(K-1) ] : nothing)
        Λ0_μx = PDMat( rand( Wishart(prior.Λ0_μx_df, inv(prior.Λ0_μx_S0)/prior.Λ0_μx_df) ) )
        μ0_μx = rand(MvNormal(prior.μ0_μx_mean, prior.μ0_μx_Cov))
        s0_δy = rand(InverseGamma(prior.s0_δy_df/2.0, prior.s0_δy_df*prior.s0_δy_s0/2.0))
        ν_δy = 5.0
        Λ0star_ηy = PDMat( rand( Wishart(prior.Λ0star_ηy_df, inv(prior.Λ0star_ηy_S0)/prior.Λ0star_ηy_df) ) )
        β0star_ηy = rand( MvNormal(prior.β0star_ηy_mean, prior.β0star_ηy_Cov) )
        α = rand(Gamma(prior.α_sh, 1.0/prior.α_rate))
        lω = log.(fill(1.0 / H, H)) # start with equal weights
        S = [ sample(Weights(ones(H))) for i = 1:n ] # random allocation

        if γglobal
            γ = trues(K) # always start with all variables
        else
            γ = trues(H, K)
        end

        γδc = Inf
        # γδc = fill(1.0e6, K)
        # γδc = nothing
        π_γ = fill(0.5, K)

        δ_x = [ rand(InverseGamma(ν_δx[k]/2.0, ν_δx[k]*s0_δx[k]/2.0)) for h=1:H, k=1:K ]

        β_x = ( K > 1 ? [ vcat( [rand(MvNormal(β0_βx[k], inv(Λ0_βx[k]))) for h = 1:H]'... ) for k = 1:(K-1) ] : nothing)

        μ_x = vcat([ rand(MvNormal(μ0_μx, inv(Λ0_μx))) for h = 1:H ]'...)
        δ_y = rand( InverseGamma(ν_δy/2.0, ν_δy*s0_δy/2.0), H )
        β_y = vcat([ rand(MvNormal(β0star_ηy[2:(K+1)], inv(Λ0star_ηy).mat[2:(K+1), 2:(K+1)])) for h = 1:H ]'...)
        μ_y = rand( Normal(β0star_ηy[1], sqrt(inv(Λ0star_ηy).mat[1,1])), H)
    else
        s0_δx = [ prior.s0_δx_s0[k] for k = 1:K ]
        ν_δx = fill(5.0, K)
        Λ0_βx = ( K > 1 ? [ inv(prior.Λ0_βx_S0[k]) for k = 1:(K-1) ] : nothing)
        β0_βx = ( K > 1 ? [ prior.β0_βx_mean[k] for k = 1:(K-1) ] : nothing)
        Λ0_μx = inv(prior.Λ0_μx_S0)
        μ0_μx = deepcopy(prior.μ0_μx_mean)
        s0_δy = deepcopy(prior.s0_δy_s0)
        ν_δy = 5.0
        Λ0star_ηy = inv(prior.Λ0star_ηy_S0)
        β0star_ηy = deepcopy(prior.β0star_ηy_mean)
        α = prior.α_sh / prior.α_rate
        lω = log.(fill(1.0 / H, H))
        S = [ sample(Weights(ones(H))) for i = 1:n ]

        if γglobal
            γ = trues(K)
        else
            γ = trues(H, K)
        end

        γδc = Inf
        # γδc = fill(1.0e6, K)
        # γδc = nothing
        π_γ = fill(0.5, K)

        δ_x = vcat([ deepcopy(s0_δx) for h = 1:H ]'...)

        β_x = ( K > 1 ? [ vcat( [deepcopy(β0_βx[k]) for h = 1:H]'... ) for k = 1:(K-1) ] : nothing)

        μ_x = vcat([ deepcopy(μ0_μx) for h = 1:H ]'...)
        δ_y = fill(s0_δy, H)
        β_y = vcat([ β0star_ηy[2:(K+1)] for h = 1:H ]'...)
        μ_y = fill(β0star_ηy[1], H)
    end

    cSig_ηlδx = [ PDMat(Matrix(Diagonal(fill(1.0, Int(K+K*(K+1)/2))))) for h = 1:H ]
    adapt = false

    State_DPmRegJoint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ,
        S, lω, α, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
        β0_βx, Λ0_βx, ν_δx, s0_δx,
        cSig_ηlδx, adapt)
end

function llik_DPmRegJoint(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::BitArray{1}, γδc::Union{Float64, Array{T, 1}},
    lω::Array{T, 1},
    lωNX_vec::Array{T, 1}) where T <: Real

    llik_num_mat = llik_numerator(y, X, K, H, μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, lω)
    return llik_DPmRegJoint(llik_num_mat, lωNX_vec)
end
function llik_DPmRegJoint(llik_numerator_mat::Array{T, 2},
    lωNX_vec::Array{T, 1}) where T <: Real

    llik_num_vec = vec( BayesInference.logsumexp(llik_numerator_mat, 2) ) # n vector
    llik_vec = llik_num_vec - lωNX_vec
    return sum(llik_vec)
end
