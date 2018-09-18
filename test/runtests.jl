using DPmRegJoint
using Test

# @test hello("Julia") == "Hello, Julia"
# @test domath(2.0) ≈ 7.0

@test mean(InverseGamma(2.0, 2.0)) ≈ 1.0 # make sure they don't change parameterization
@test mean(Gamma(2.0, 2.0)) ≈ 4.0 # make sure they don't change parameterization

n = 500
K = 10
H = 50

y = randn(n)
X = randn(n, K)

prior = Prior_DPmRegJoint(1.0, # α_sh
1.0, # α_rate
randn(K+1), # β0star_ηy_mean
PDMat(Matrix(Diagonal(fill(10.0, K+1)))), # β0star_ηy_Cov
2.0*K, # Λ0star_ηy_df
PDMat(Matrix(Diagonal(fill(10.0, K+1)))), # Λ0star_ηy_S0
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

state = State_DPmRegJoint(randn(H), # μ_y,
randn(H, K), # β_y,
exp.(randn(H)), # δ_y,
randn(H, K), # μ_x,
[ randn(H, k) for k = (K-1):-1:1 ], # β_x,
exp.(randn(H, K)), # δ_x,
[ sample(Weights(ones(H))) for i = 1:n ], # S,
rDirichlet(ones(H), true), # lω,
1.0, # α,
randn(K+1), # β0star_ηy,
PDMat(Matrix(Diagonal(ones(K+1)))), # Λ0star_ηy,
5.0, # ν_δy,
1.0, # s0_δy,
randn(K), # μ0_μx,
PDMat(Matrix(Diagonal(ones(K)))), # Λ0_μx,
[ randn(k) for k = (K-1):-1:1 ], # β0_βx,
[ PDMat(Matrix(Diagonal(fill(10.0, k)))) for k = (K-1):-1:1 ], # Λ0_βx,
fill(5.0, K), # ν_δx,
ones(K), # s0_δx
[ PDMat(Matrix(Diagonal(fill(0.5, Int(K + K*(K+1)/2))))) for h = 1:H ], # cSig_ηx,
true # adapt
)


model = Model_DPmRegJoint(y, # y,
X, # X,
H, # H,
prior, # prior,
state # state
)

model.state.lNX = lNXmat(model.X, model.state.μ_x, model.state.β_x, model.state.δ_x)
@time lNXmat(model.X, model.state.μ_x, model.state.β_x, model.state.δ_x)
model.state.lωNX_vec = lωNXvec(model.state.lω, model.state.lNX)

@time update_alloc!(model, hcat(model.y, model.X))

ω_start = [0.2, 0.3, 0.25, 0.15, 0.10]
v = lω_to_v(log.(ω_start))

for i = 1:1000
    global lω = v_to_lω(v)
    global v = lω_to_v(lω)
end
exp.(lω) ≈ ω_start

@time update_vlω_mvSlice!(model)

Λβ0star_ηy = model.state.Λ0star_ηy * model.state.β0star_ηy
βΛβ0star_ηy = PDMats.quad(model.state.Λ0star_ηy, model.state.β0star_ηy)

h = 7
@time update_η_h_Met!(model, h, Λβ0star_ηy, βΛβ0star_ηy)

@time mcmc_DPmRegJoint!(model, 200, Monitor_DPmRegJoint(false, false, false), "out_progress.txt", 1, 10)
rm("out_progress.txt")

println(model.state.iter)
println(model.state.accpt / float(model.state.iter))
println(counts(model.state.S, 1:model.H))
println(exp.(model.state.lω))
println(model.state.α)
