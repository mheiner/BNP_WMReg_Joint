module DPmRegJoint

using Dates
using LinearAlgebra
using SpecialFunctions

using Distributions
using PDMats
using StatsBase

using BayesInference # personal package

include("general.jl")
include("mcmc.jl")
include("sqfChol.jl")
include("update_alloc.jl")
include("update_weights.jl")
include("update_eta_Met.jl")
include("update_G0.jl")
include("update_variable_selection.jl")
include("densityEstimation.jl")

end
