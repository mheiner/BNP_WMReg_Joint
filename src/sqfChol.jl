# sqfChol.jl

export sqfChol_to_Σ, PDMat_adj;

## Note this follows my unconventional definition of the square-root-free
# Cholesky decomposition with construction from the back of the vector.
function sqfChol_to_Σ(β::Array{Array{T, 1}, 1}, δ::Array{T, 1},
    pdthrowerror::Bool=true) where T <: Real

    sqrtΔ = Diagonal(sqrt.(δ))
    βmat = BayesInference.xpnd_tri(vcat(β...), false)'

    βinvSqrtΔ = Matrix(βmat \ sqrtΔ)
    Σ = βinvSqrtΔ * βinvSqrtΔ'

    PDMat_adj(Σ, throwerror=pdthrowerror)
end

function lNX_sqfChol(X::Union{Array{T, 1}, Array{T, 2}}, μ::Array{T, 1},
    β::Array{Array{T, 1}, 1}, δ::Array{T, 1}, pdthrowerror::Bool=true) where T <: Real

    size(X,1) == length(μ) || throw("In lNX_sqfChol, the columns of X are the observations.")

    Σ = sqfChol_to_Σ(β, δ, pdthrowerror)

    if isnothing(Σ)
        return nothing
    else
        d = MultivariateNormal(μ, Σ)
        if pdthrowerror
            return logpdf(d, X)
        else
            try logpdf(d, X)
            catch excep
                if isa(excep, SingularException)
                    println("Singular exception in lNX_sqfChol.\n")
                    return nothing
                else
                    return logpdf(d, X)
                end
            end
        end
    end
end

# function lNX_seq(x::Array{T, 1}, μ::Array{T, 1},
#     β::Array{Array{T, 1}, 1}, δ::Array{T, 1}) where T <: Real
#
#     K = length(μ)
#     dev = x .- μ
#
#     out = ldnorm(x[K], μ[K], δ[K])
#
#     for k = (K-1):-1:1
#         out += ldnorm(x[k], μ[k] - sum(β[k] .* dev[(k+1):K]) , δ[k])
#     end
#
#     out
# end

### test
#
# using BayesInference
# using Distributions
# using PDMats
# using LinearAlgebra
#
# n = 1000
# K = 5
# X = randn(n, K)
#
# μ_x = randn(K)
# δ_x = exp.(randn(K))
# β_x = [ randn(k) for k = (K-1):-1:1 ]
#
#
# Σ = sqfChol_to_Σ(β_x, δ_x)
# MvNormal(μ_x, Σ)
#
# using BenchmarkTools
#
# @time out1 = lNX(Array(X'), μ_x, β_x, δ_x)
# @btime lNX(Array(X'), μ_x, β_x, δ_x)
# # lNX(X[1,:], μ_x, β_x, δ_x)
#
# @time out2 = [ lNX_seq(X[i,:], μ_x, β_x, δ_x) for i = 1:size(X, 1) ]
# @btime [ lNX_seq(X[i,:], μ_x, β_x, δ_x) for i = 1:size(X, 1) ]
#
# out1 ≈ out2

# naive implemtation (non-sequential) is far superior

function PDMat_adj(A::Matrix{Float64}; maxadd::Float64=1.0e-6,
        epsfact::Float64=100.0, cumadd::Float64=0.0, throwerror::Bool=true)

    try PDMat(A)
    catch excep
        if isa(excep, PosDefException) && cumadd <= maxadd
            a = epsfact * eps(Float64)
            A += a * I
            cumadd += a
            epsfactnext = 10.0 * epsfact
            return PDMat_adj(A, maxadd=maxadd, epsfact=epsfactnext,
                    cumadd=cumadd, throwerror=throwerror)
        else
            println("Failure to adjust to positive definiteness.\n A = ", A, "\n")
            if throwerror
                return PDMat(A) # just trigger original error
            else
                return nothing
            end
        end
    end
end
