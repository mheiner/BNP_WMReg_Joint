module DPmRegJoint

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


end
