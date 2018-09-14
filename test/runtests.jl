using DPmRegJoint
using Test

# @test hello("Julia") == "Hello, Julia"
# @test domath(2.0) ≈ 7.0

n = 100
K = 5
H = 25

y = randn(n)
X = randn(n, K)

prior = Prior_DPmRegJoint(1.0, # α_sh
1.0, # α_rate
randn(K+1), # β0_ηy_mean
PDMat(Matrix(Diagonal(fill(10.0, K+1)))), # β0_ηy_Cov
2.0*K, # Λ0_ηy_df
PDMat(Matrix(Diagonal(fill(10.0, K+1)))), # Λ0_ηy_S0
5.0, # s0_δy_df
1.0, # s0_δy_s0
randn(K+1), # μ0_μx_mean
PDMat(Matrix(Diagonal(fill(10.0, K)))), # μ0_μx_Cov
2.0*K, # Λ0_μx_df
PDMat(Matrix(Diagonal(fill(10.0, K)))), # Λ0_μx_S0
[randn(k) for k = (K-1):-1:1], # β0_βx_mean
[ PDMat(Matrix(Diagonal(fill(10.0, k)))) for k = (K-1):-1:1 ], # β0_βx_Cov
fill(2.0*K, K-1), # Λ0_βx_df
[ PDMat(Matrix(Diagonal(fill(10.0, k)))) for k = (K-1):-1:1 ], # Λ0_βx_S0
fill(5.0, K), # s0_δx_df
fill(1.0, K)) # s0_δx_s0

state = Params_DPmRegJoint(randn(H), # μ_y,
randn(H, K), # β_y,
exp.(randn(H)), # δ_y,
randn(H, K), # μ_x,
[ randn(H, k) for k = (K-1):-1:1 ], # β_x,
exp.(randn(H, K)), # δ_x,
[ sample(Weights(ones(H))) for i = 1:n ], # S,
rDirichlet(ones(H), false), # ω,
1.0, # α,
randn(K+1), # β0_ηy,
PDMat(Matrix(Diagonal(ones(K+1)))), # Λ0_ηy,
5.0, # ν_δy,
1.0, # s0_δy,
randn(K), # μ0_μx,
PDMat(Matrix(Diagonal(ones(K)))), # Λ0_μx,
[ randn(k) for k = (K-1):-1:1 ], # β0_βx,
[ PDMat(Matrix(Diagonal(fill(10.0, k)))) for k = (K-1):-1:1 ], # Λ0_βx,
fill(5.0, K), # ν_δx,
ones(K), # s0_δx
)


model = Model_DPmRegJoint(y, # y,
X, # X,
H, # H,
prior, # prior,
state, # state,
0, # iter,
zeros(Int, H), # accpt,
[ PDMat(Matrix(Diagonal(fill(0.5, Int(K + K*(K+1)/2))))) for h = 1:H ], # cSig_ηx,
false, # adapt,
0, # adapt_iter,
zeros(H, Int(K + K*(K+1)/2)), # runningsum_ηx,
zeros(H, Int(K + K*(K+1)/2), Int(K + K*(K+1)/2)) # runningSS_ηx
)

update_alloc!(model, hcat(model.y, model.X))
