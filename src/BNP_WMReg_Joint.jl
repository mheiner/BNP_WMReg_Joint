module BNP_WMReg_Joint

using Dates
using LinearAlgebra
using SpecialFunctions

using Distributions
using PDMats
using StatsBase
using Roots

using BayesInference # personal package

include("general.jl")
include("mcmc.jl")
include("sqfChol.jl")
include("update_alloc.jl")
include("update_weights.jl")
include("update_eta_Met.jl")
include("update_G0.jl")
include("update_variable_selection_local.jl")
include("update_variable_selection_global.jl")
include("densityEstimation.jl")

end
