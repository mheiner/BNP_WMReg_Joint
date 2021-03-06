# general.jl

export State_BNP_WMReg_Joint, init_state_BNP_WMReg_Joint,
    Prior_BNP_WMReg_Joint, Model_BNP_WMReg_Joint,
    Monitor_BNP_WMReg_Joint, Updatevars_BNP_WMReg_Joint,
    lNXmat, reset_adapt!,
    llik_BNP_WMReg_Joint;

mutable struct State_BNP_WMReg_Joint

    ### Parameters
    # component (kernel parameters) η
    μ_y::Array{Float64, 1}  # H vector
    β_y::Array{Float64, 2}  # H by K matrix
    δ_y::Array{Float64, 1}  # H vector
    μ_x::Array{Float64, 2}  # H by K matrix
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing}  # vector of H by (k in (K-1):1) matrices
    δ_x::Array{Float64,2}   # H by K matrix

    # variable selection
    γ::Union{BitArray{1}, BitArray{2}} # K vector (for global, H by K matrix for local)
    γδc::Union{Float64, Array{Float64, 1}, Nothing} # scale factors for δ_x
    π_γ::Array{Float64, 1} # K vector of probabilities that γ_k = 1 (true in code)
    ξ::BitArray{1} # mixture indicator for pi_gamma

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
    adapt_thin::Union{Int, Nothing}
    runningsum_ηlδx::Union{Array{Float64, 2}, Nothing} # H by (K + K(K+1)/2) matrix (unless using Sigma_x_diag, in which case it has 2K columns)
    runningSS_ηlδx::Union{Array{Float64, 3}, Nothing}  # H by (K + K(K+1)/2) by (K + K(K+1)/2) matrix

    lNX::Array{Float64, 2} # n by H matrix of log(pdf(Normal(x_i))) under each obs i and allocation h
    lωNX_vec::Array{Float64, 1} # n vector with log( sum_j ωN(x_i) )

    n_occup::Int
    llik::Float64

end

### Outer constructors for State_BNP_WMReg_Joint
## for coninuing an adapt phase
function State_BNP_WMReg_Joint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ, ξ,
    S, lω::Vector{Float64}, α::Float64, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx,
    adapt, adapt_iter, adapt_thin, runningsum_ηlδx, runningSS_ηlδx, lNX, lωNX_vec, llik)

    State_BNP_WMReg_Joint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ), deepcopy(ξ),
        deepcopy(S), deepcopy(lω), lω_to_v(lω), α, deepcopy(β0star_ηy),
        deepcopy(Λ0star_ηy), ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        iter, accpt, deepcopy(cSig_ηlδx),
        adapt, adapt_iter, adapt_thin, deepcopy(runningsum_ηlδx), deepcopy(runningSS_ηlδx),
        deepcopy(lNX), deepcopy(lωNX_vec), length(unique(S)), llik)
end
## same but includes v
function State_BNP_WMReg_Joint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ, ξ,
    S, lω::Vector{Float64}, v::Vector{Float64}, α::Float64, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx,
    adapt, adapt_iter, adapt_thin, runningsum_ηlδx, runningSS_ηlδx, lNX, lωNX_vec, llik)

    State_BNP_WMReg_Joint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ), deepcopy(ξ),
        deepcopy(S), deepcopy(lω), v, α, deepcopy(β0star_ηy),
        deepcopy(Λ0star_ηy), ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        iter, accpt, deepcopy(cSig_ηlδx),
        adapt, adapt_iter, adapt_thin, deepcopy(runningsum_ηlδx), deepcopy(runningSS_ηlδx),
        deepcopy(lNX), deepcopy(lωNX_vec), length(unique(S)), llik)
end

## for starting new
function State_BNP_WMReg_Joint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ, ξ,
    S, lω::Vector{Float64}, α::Float64, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    cSig_ηlδx, adapt)

    State_BNP_WMReg_Joint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ), deepcopy(ξ),
        deepcopy(S), deepcopy(lω), lω_to_v(lω), α, deepcopy(β0star_ηy), deepcopy(Λ0star_ηy),
        ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        0, zeros(Int, length(lω)), deepcopy(cSig_ηlδx),
        adapt, 0, 1,
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2),
                        Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros(Float64, 1, 1), zeros(Float64, 1),
        length(unique(S)), 0.0 )
end

## for starting new ( given a value of v )
function State_BNP_WMReg_Joint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ, ξ,
    S, lω::Vector{Float64}, v::Vector{Float64}, α, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    cSig_ηlδx, adapt)

    State_BNP_WMReg_Joint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ), deepcopy(ξ),
        deepcopy(S), deepcopy(lω), deepcopy(v), α, deepcopy(β0star_ηy), deepcopy(Λ0star_ηy),
        ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        0, zeros(Int, length(lω)), deepcopy(cSig_ηlδx),
        adapt, 0, 1,
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros( Float64, length(lω), Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2),
                        Int(length(μ0_μx) + length(μ0_μx)*(length(μ0_μx)+1)/2) ),
        zeros(Float64, 1, 1), zeros(Float64, 1),
        length(unique(S)), 0.0 )
end

## for starting new but not adapting
function State_BNP_WMReg_Joint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ, ξ,
    S, lω::Vector{Float64}, α::Float64, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
    β0_βx, Λ0_βx, ν_δx, s0_δx,
    iter, accpt, cSig_ηlδx)

    State_BNP_WMReg_Joint(deepcopy(μ_y), deepcopy(β_y), deepcopy(δ_y), deepcopy(μ_x),
        deepcopy(β_x), deepcopy(δ_x), deepcopy(γ), deepcopy(γδc), deepcopy(π_γ), deepcopy(ξ),
        deepcopy(S), deepcopy(lω), lω_to_v(lω), α, deepcopy(β0star_ηy),
        deepcopy(Λ0star_ηy), ν_δy, s0_δy, deepcopy(μ0_μx), deepcopy(Λ0_μx),
        deepcopy(β0_βx), deepcopy(Λ0_βx), deepcopy(ν_δx), deepcopy(s0_δx),
        iter, accpt, deepcopy(cSig_ηlδx),
        false, nothing, nothing, nothing, nothing,
        zeros(Float64, 1, 1), zeros(Float64, 1),
        length(unique(S)), 0.0)
end

mutable struct Prior_BNP_WMReg_Joint
    α_sh::Float64   # gamma shape
    α_rate::Float64 # gamma rate

    π_sh::Array{Float64, 2} # Beta shape parameters
    π_ξ::Union{Float64, Array{Float64, 1}} # Mixture probability that pi_gamma comes from a beta (instead of point mass at 0)

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

### Outer constructors for Prior_BNP_WMReg_Joint
## automatic creation of precision matrices from covariance matrices
function Prior_BNP_WMReg_Joint(α_sh, α_rate, π_sh, π_ξ, β0star_ηy_mean, β0star_ηy_Cov,
    Λ0star_ηy_df, Λ0star_ηy_S0, s0_δy_df, s0_δy_s0,
    μ0_μx_mean, μ0_μx_Cov, Λ0_μx_df, Λ0_μx_S0,
    β0_βx_mean, β0_βx_Cov, Λ0_βx_df, Λ0_βx_S0,
    s0_δx_df, s0_δx_s0)

    Prior_BNP_WMReg_Joint(α_sh, α_rate, deepcopy(π_sh), deepcopy(π_ξ),
        deepcopy(β0star_ηy_mean),
        deepcopy(β0star_ηy_Cov), inv(β0star_ηy_Cov),
        deepcopy(Λ0star_ηy_df), deepcopy(Λ0star_ηy_S0), s0_δy_df, s0_δy_s0,
        deepcopy(μ0_μx_mean), deepcopy(μ0_μx_Cov), inv(μ0_μx_Cov), Λ0_μx_df,
        deepcopy(Λ0_μx_S0), deepcopy(β0_βx_mean), deepcopy(β0_βx_Cov),
        ( typeof(β0_βx_Cov) == Nothing ? nothing : [inv(β0_βx_Cov[k]) for k = 1:length(β0_βx_Cov)] ),
        deepcopy(Λ0_βx_df), deepcopy(Λ0_βx_S0),
        deepcopy(s0_δx_df), deepcopy(s0_δx_s0))
end

## default prior spec
function Prior_BNP_WMReg_Joint(K::Int, H::Int;
    center_y::Float64=0.0, center_X::Vector{Float64}=zeros(K),
    range_y::Float64=6.0, range_X::Vector{Float64}=fill(6.0, K), snr::Float64=5.0, 
    Σx_type::Symbol=:full, γ_type::Symbol=:global) # the default 6.0 assumes data standardized

    s0_δy_s0 = (range_y/6.0)^2 / snr

    if Σx_type == :full
        s0_δx_s0_ranXfact = 8.0
        α_sh = 5.0
    elseif Σx_type == :diag
        s0_δx_s0_ranXfact = 12.0
        α_sh = 6.0
    end
    
    s0_δx_s0 = (range_X ./ s0_δx_s0_ranXfact ).^2


    Prior_BNP_WMReg_Joint(α_sh, # α_sh
    1.0, # α_rate
    hcat(fill(1.0, K), fill(0.5, K)), # π_sh
    fill(0.25, K), # π_ξ
    vcat(center_y, zeros(K)), # β0star_ηy_mean
    PDMat(Matrix(Diagonal(vcat((range_y/6.0)^2, fill(1.0, K))))), # β0star_ηy_Cov
    50.0*(K+1+2), # Λ0star_ηy_df # was multiplied by 100...
    PDMat(Matrix(Diagonal( vcat((range_y/2.0)^2, fill(16.0, K)) ./ s0_δy_s0 ))), # Λ0star_ηy_S0
    5.0, # s0_δy_df
    s0_δy_s0, # s0_δy_s0; the outer divide follows a SNR arguement
    center_X, # μ0_μx_mean
    PDMat(Matrix(Diagonal( (range_X ./ 6.0).^2 ))), # μ0_μx_Cov; needs to stay in close to center_X
    10.0*(K+2), # Λ0_μx_df; should be strong
    PDMat(Matrix(Diagonal( (range_X ./ 2.0).^2  ))), # Λ0_μx_S0; should give flexibility to mu_xs
    (K > 1 ? [zeros(k) for k = (K-1):-1:1] : nothing), # β0_βx_mean
    (K > 1 ? [ PDMat(Matrix(Diagonal(fill(1.0, k)))) for k = (K-1):-1:1 ] : nothing), # β0_βx_Cov
    (K > 1 ? fill(10.0*(K+2), K-1) : nothing), # Λ0_βx_df
    (K > 1 ? [ PDMat(Matrix(Diagonal(fill(2.0, k)))) for k = (K-1):-1:1 ] : nothing), # Λ0_βx_S0
    fill(5.0, K), # s0_δx_df
    s0_δx_s0 ) # s0_δx_s0
end



mutable struct Model_BNP_WMReg_Joint
    y::Array{Float64, 1}
    X::Array{Float64, 2}
    n::Int # length of data
    K::Int # number of predictors (columns in X)
    H::Int # DP truncation level
    prior::Prior_BNP_WMReg_Joint

    indx_ηy::Dict
    indx_ηx::Dict
    indx_β_x::Union{Dict, Nothing}

    Σx_type::Symbol # currently one of :full and :diag
    γ_type::Symbol # will one of :global, :local, or :fixed (for no variable selection)

    state::State_BNP_WMReg_Joint # this is the only thing that should change

    Model_BNP_WMReg_Joint(y, X, n, K, H, prior,
        indx_ηy, indx_ηx, indx_β_x, Σx_type, γ_type, state) = new(deepcopy(y), deepcopy(X), copy(n),
        deepcopy(K), deepcopy(H), deepcopy(prior),
        deepcopy(indx_ηy), deepcopy(indx_ηx), deepcopy(indx_β_x),
        deepcopy(Σx_type), deepcopy(γ_type),
        deepcopy(state))
end

function Model_BNP_WMReg_Joint(y::Array{Float64, 1}, X::Array{Float64, 2},
    H::Int, prior::Prior_BNP_WMReg_Joint, state::State_BNP_WMReg_Joint; Σx_type::Symbol=:full, γ_type::Symbol=:global)

    n = length(y)
    nx, K = size(X)

    n == nx || throw(error("X and y size mismatch."))

    indx_ηy = Dict( :μ => 1, :β => 2:(K+1), :δ => K+2 )

    if K > 1 && Σx_type == :full
        indx_ηx = Dict( :μ => 1:K, :β => (K+1):Int(K*(K+1)/2), :δ => Int(K*(K+1)/2 + 1):Int(K*(K+1)/2 + K) )
        indx_β_x = Dict()
        ii = 1
        jj = K - 1
        for k = 1:(K-1)
            indx_β_x[k] = Int(ii):Int(ii+jj-1)
            ii += jj
            jj -= 1
        end
    elseif K > 1 && Σx_type == :diag
        indx_ηx = Dict( :μ => 1:K, :δ => Int(K+1):Int(2*K) )
        indx_β_x = nothing
        state.β_x = nothing
        state.β0_βx = nothing
        state.Λ0_βx = nothing
        size(state.cSig_ηlδx[1], 1) == 2*K || throw("Initialize cSig_ηlδx with correct dimensions.")
        state.runningsum_ηlδx = zeros(Float64, H, 2*K)
        state.runningSS_ηlδx = zeros(Float64, H, 2*K, 2*K)
    else
        indx_ηx = Dict( :μ => 1:K, :δ => Int(K+1):Int(2*K) )
        indx_β_x = nothing
    end

    return Model_BNP_WMReg_Joint(deepcopy(y), deepcopy(X), n, K, deepcopy(H), deepcopy(prior),
        indx_ηy, indx_ηx, indx_β_x, Σx_type, γ_type, deepcopy(state))
end

mutable struct Monitor_BNP_WMReg_Joint
    ηlω::Bool
    γ::Bool
    S::Bool
    G0::Bool
end

mutable struct Updatevars_BNP_WMReg_Joint
    η::Bool
    lω::Bool
    γ::Bool
    α::Bool
    S::Bool
    G0::Bool
end

function postSimsInit_BNP_WMReg_Joint(m::Monitor_BNP_WMReg_Joint, n_keep::Int, init_state::State_BNP_WMReg_Joint)

    symb = [ :n_occup, :llik ]

    if m.ηlω
        push!(symb, [:μ_y, :β_y, :δ_y, :μ_x, :β_x, :δ_x, :lω, :α]...)
    end

    if m.γ
        push!(symb, [:γ, :π_γ, :ξ]...)
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

function lNXmat_Σdiag(X::Array{T,2}, μ::Array{T,2}, δ::Array{T,2}) where T <: Real
    n = size(X,1)
    H, KK = size(μ)
    out = zeros(n, H)

    for i = 1:n
        for h = 1:H
            for k = 1:KK
                out[i,h] += -0.5*log( 2.0π * δ[h,k] ) - 0.5*( X[i,k] - μ[h,k] )^2 / δ[h,k]
            end
        end
    end

    return out
end

function lNX_Σdiag(X::Union{Array{T, 1}, Array{T, 2}}, μ::Array{T, 1}, δ::Array{T, 1}) where T <: Real

    size(X,1) == length(μ) || throw("In lNX_sqfChol functions, the columns of X are the observations.")

    KK, n = size(X)

    out = zeros(n)

    for i = 1:n
        for k = 1:KK
            out[i] += -0.5*log( 2.0π * δ[k] ) - 0.5*( X[k,i] - μ[k] )^2 / δ[k]
        end
    end

    return out
end


function lωNXvec(lω::Array{T, 1}, lNX_mat::Array{T, 2}) where T <: Real
    lωNX_mat = broadcast(+, permutedims(lω), lNX_mat) # n by H
    vec( BayesInference.logsumexp(lωNX_mat, 2) ) # n vector
end

function reset_adapt!(model::Model_BNP_WMReg_Joint)
    model.state.adapt_iter = 0
    model.state.runningsum_ηlδx = zeros( Float64, size(model.state.runningsum_ηlδx) )
    model.state.runningSS_ηlδx = zeros( Float64, size(model.state.runningSS_ηlδx) )
    return nothing
end

function init_state_BNP_WMReg_Joint(n::Int, K::Int, H::Int,
    prior::Prior_BNP_WMReg_Joint; random::Int=1, Σx_type::Symbol=:full, γ_type::Symbol=:global)

    ν_δx = fill(5.0, K)
    ν_δy = 5.0

    γδc = Inf
    # γδc = fill(1.0e6, K)
    # γδc = nothing
    π_γ = fill(0.25, K)
    ξ = trues(K)

    if γ_type in (:fixed, :global)
        γ = trues(K) # always start with all variables
    elseif γ_type == :local
        γ = trues(H, K)
    end

    if random == 0
        s0_δx = [ prior.s0_δx_s0[k] for k = 1:K ]
        Λ0_βx = ( K > 1 ? [ inv(prior.Λ0_βx_S0[k]) for k = 1:(K-1) ] : nothing)
        β0_βx = ( K > 1 ? [ prior.β0_βx_mean[k] for k = 1:(K-1) ] : nothing)
        Λ0_μx = inv(prior.Λ0_μx_S0)
        μ0_μx = deepcopy(prior.μ0_μx_mean)
        s0_δy = deepcopy(prior.s0_δy_s0)
        Λ0star_ηy = inv(prior.Λ0star_ηy_S0)
        β0star_ηy = deepcopy(prior.β0star_ηy_mean)
        α = prior.α_sh / prior.α_rate
        lω = log.(fill(1.0 / H, H))
        v = lω_to_v(lω)
        S = [ sample(Weights(ones(H))) for i = 1:n ]
        δ_x = vcat([ deepcopy(s0_δx) for h = 1:H ]'...)
        β_x = ( K > 1 ? [ vcat( [deepcopy(β0_βx[k]) for h = 1:H]'... ) for k = 1:(K-1) ] : nothing)
        μ_x = vcat([ deepcopy(μ0_μx) for h = 1:H ]'...)
        δ_y = fill(s0_δy, H)
        β_y = vcat([ β0star_ηy[2:(K+1)] for h = 1:H ]'...)
        μ_y = fill(β0star_ηy[1], H)
    elseif random == 1 # G0 not random
        s0_δx = [ prior.s0_δx_s0[k] for k = 1:K ]
        Λ0_βx = ( K > 1 ? [ inv(prior.Λ0_βx_S0[k]) for k = 1:(K-1) ] : nothing)
        β0_βx = ( K > 1 ? [ prior.β0_βx_mean[k] for k = 1:(K-1) ] : nothing)
        Λ0_μx = inv(prior.Λ0_μx_S0)
        μ0_μx = deepcopy(prior.μ0_μx_mean)
        s0_δy = deepcopy(prior.s0_δy_s0)
        Λ0star_ηy = inv(prior.Λ0star_ηy_S0)
        β0star_ηy = deepcopy(prior.β0star_ηy_mean)
        α = rand(Gamma(prior.α_sh, 1.0/prior.α_rate))
        # lω = log.(fill(1.0 / H, H)) # start with equal weights
        lω, lv = rGenDirichlet(ones(H-1), fill(α, H-1); logout=true)
        v = exp.(lv)
        S = [ sample(Weights(ones(H))) for i = 1:n ] # random allocation
        δ_x = [ rand(InverseGamma(ν_δx[k]/2.0, ν_δx[k]*s0_δx[k]/2.0)) for h=1:H, k=1:K ]
        β_x = ( K > 1 ? [ vcat( [rand(MvNormal(β0_βx[k], inv(Λ0_βx[k]))) for h = 1:H]'... ) for k = 1:(K-1) ] : nothing)
        μ_x = vcat([ rand(MvNormal(μ0_μx, inv(Λ0_μx))) for h = 1:H ]'...)
        δ_y = rand( InverseGamma(ν_δy/2.0, ν_δy*s0_δy/2.0), H )
        β_y = vcat([ rand(MvNormal(β0star_ηy[2:(K+1)], δ_y[h]*inv(Λ0star_ηy).mat[2:(K+1), 2:(K+1)])) for h = 1:H ]'...)
        μ_y = [ rand( Normal(β0star_ηy[1], sqrt(δ_y[h]*inv(Λ0star_ηy).mat[1,1])) ) for h = 1:H ]
    elseif random == 2
        s0_δx = [ rand(Gamma(ν_δx[k]*prior.s0_δx_df[k]/2.0, 2.0*prior.s0_δx_s0[k]/(ν_δx[k]*prior.s0_δx_df[k]))) for k = 1:K ] # shape and scale
        Λ0_βx = ( K > 1 ? [ PDMat(rand(Wishart(prior.Λ0_βx_df[k], inv(prior.Λ0_βx_S0[k])/prior.Λ0_βx_df[k]))) for k = 1:(K-1) ] : nothing)
        β0_βx = ( K > 1 ? [ rand(MvNormal(prior.β0_βx_mean[k], prior.β0_βx_Cov[k])) for k = 1:(K-1) ] : nothing)
        Λ0_μx = PDMat( rand( Wishart(prior.Λ0_μx_df, inv(prior.Λ0_μx_S0)/prior.Λ0_μx_df) ) )
        μ0_μx = rand(MvNormal(prior.μ0_μx_mean, prior.μ0_μx_Cov))
        s0_δy = rand(Gamma(ν_δy*prior.s0_δy_df/2.0, 2.0*prior.s0_δy_s0/(ν_δy*prior.s0_δy_df))) # the prior here was actually gamma
        Λ0star_ηy = PDMat( rand( Wishart(prior.Λ0star_ηy_df, inv(prior.Λ0star_ηy_S0)/prior.Λ0star_ηy_df) ) )
        β0star_ηy = rand( MvNormal(prior.β0star_ηy_mean, prior.β0star_ηy_Cov) )
        α = rand(Gamma(prior.α_sh, 1.0/prior.α_rate))
        # lω = log.(fill(1.0 / H, H)) # start with equal weights
        lω, lv = rGenDirichlet(ones(H-1), fill(α, H-1); logout=true)
        v = exp.(lv)
        S = [ sample(Weights(ones(H))) for i = 1:n ] # random allocation
        δ_x = [ rand(InverseGamma(ν_δx[k]/2.0, ν_δx[k]*s0_δx[k]/2.0)) for h=1:H, k=1:K ]
        β_x = ( K > 1 ? [ vcat( [rand(MvNormal(β0_βx[k], inv(Λ0_βx[k]))) for h = 1:H]'... ) for k = 1:(K-1) ] : nothing)
        μ_x = vcat([ rand(MvNormal(μ0_μx, inv(Λ0_μx))) for h = 1:H ]'...)
        δ_y = rand( InverseGamma(ν_δy/2.0, ν_δy*s0_δy/2.0), H )
        β_y = vcat([ rand(MvNormal(β0star_ηy[2:(K+1)], δ_y[h]*inv(Λ0star_ηy).mat[2:(K+1), 2:(K+1)])) for h = 1:H ]'...)
        μ_y = [ rand( Normal(β0star_ηy[1], sqrt(δ_y[h]*inv(Λ0star_ηy).mat[1,1]))) for h = 1:H ]
    end

    if Σx_type == :full
        cSig_ηlδx = [ PDMat(Matrix(Diagonal(fill(0.1, Int(K+K*(K+1)/2))))) for h = 1:H ]
    elseif Σx_type == :diag
        cSig_ηlδx = [ PDMat(Matrix(Diagonal(fill(0.1, Int(2*K))))) for h = 1:H ]
        β_x = nothing
    end

    adapt = false

    State_BNP_WMReg_Joint(μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, π_γ, ξ,
        S, lω, v, α, β0star_ηy, Λ0star_ηy, ν_δy, s0_δy, μ0_μx, Λ0_μx,
        β0_βx, Λ0_βx, ν_δx, s0_δx,
        cSig_ηlδx, adapt)
end


function llik_BNP_WMReg_Joint(model::Model_BNP_WMReg_Joint)
    llik_num_mat = llik_numerator(model)
    return llik_BNP_WMReg_Joint(llik_num_mat, model.state.lωNX_vec)
end
function llik_BNP_WMReg_Joint(y::Array{T,1}, X::Array{T,2}, K::Int, H::Int,
    μ_y::Array{T, 1}, β_y::Array{T, 2}, δ_y::Array{T, 1},
    μ_x::Array{T, 2},
    β_x::Union{Array{Array{Float64, 2}, 1}, Nothing},
    δ_x::Array{Float64,2},
    γ::Union{BitArray{1}, BitArray{2}}, γδc::Union{Float64, Array{T, 1}},
    lω::Array{T, 1},
    lωNX_vec::Array{T, 1}) where T <: Real

    llik_num_mat = llik_numerator(y, X, K, H, μ_y, β_y, δ_y, μ_x, β_x, δ_x, γ, γδc, lω)
    return llik_BNP_WMReg_Joint(llik_num_mat, lωNX_vec)
end
function llik_BNP_WMReg_Joint(llik_numerator_mat::Array{T, 2},
    lωNX_vec::Array{T, 1}) where T <: Real

    llik_num_vec = vec( BayesInference.logsumexp(llik_numerator_mat, 2) ) # n vector
    llik_vec = llik_num_vec - lωNX_vec
    return sum(llik_vec)
end


"""
    rGenDirichlet(a, b[, logout])

  Single draw from the generalized Dirichlet distribution (Connor and Mosimann '69), option for log scale. From SparseProbVec package.

  ### Example
  ```julia
  a = ones(5)
  b = ones(5)
  rGenDirichlet(a, b)
  ```
"""
function rGenDirichlet(a::Vector{Float64}, b::Vector{Float64}; logout::Bool=false)
    n = length(a)
    K = n + 1
    length(b) == n || error("Dimension mismatch between a and b.")
    all(a .> 0.0) || error("All elements of a must be positive.")
    all(b .> 0.0) || error("All elements of b must be positive.")

    lz = Vector{Float64}(undef, n)
    loneminusz = Vector{Float64}(undef, n)
    for i = 1:n
        lx1 = log( rand( Distributions.Gamma( a[i] ) ) )
        lx2 = log( rand( Distributions.Gamma( b[i] ) ) )
        lxm = max(lx1, lx2)
        lxsum = lxm + log( exp(lx1 - lxm) + exp(lx2 - lxm) ) # logsumexp
        lz[i] = lx1 - lxsum
        loneminusz[i] = lx2 - lxsum
    end

    ## break the Stick
    lw = Vector{Float64}(undef, K)
    lwhatsleft = 0.0

    for i in 1:n
        lw[i] = lz[i] + lwhatsleft
        # lwhatsleft += log( 1.0 - exp(lw[i] - lwhatsleft) ) # logsumexp (not numerically stable)
        lwhatsleft += loneminusz[i]
    end
    lw[K] = copy(lwhatsleft)

    if logout
        out = (lw, lz)
    else
        out = (exp.(lw), exp.(lz))
    end
    return out
end
