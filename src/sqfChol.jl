# sqfChol.jl

export sqfChol2Σ;

## Note this follows my unconventional definition of the square-root-free
# Cholesky decomposition with construction from the back of the vector.
function sqfChol2Σ(β::Array{Array{T, 1}, 1}, δ::Array{T, 1}) where T <: Real
    sqrtΔ = Diagonal(sqrt.(δ_x))
    βmat = BayesInference.xpnd_tri(vcat(β_x...), false)'

    βinvSqrtΔ = Matrix(βmat \ sqrtΔ)
    Σ = βinvSqrtΔ * βinvSqrtΔ'

    PDMat(Σ)
end

## This function doesn't need to be exported. Just use its contents in the code to be readable.
# function lNX(X::Union{Array{T, 1}, Array{T, 2}}, μ::Array{T, 1},
#     β::Array{Array{T, 1}, 1}, δ::Array{T, 1}) where T <: Real
#
#     Σ = sqfChol2Σ(β, δ)
#     d = MultivariateNormal(μ, Σ)
#
#     logpdf(d, X)
# end

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
# Σ = sqfChol2Σ(β_x, δ_x)
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
