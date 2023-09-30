using Test, LinearAlgebra, SparseArrays, BandedMatrices, ILUZero, IncompleteLU

println("Testing...")

include("test_Diagonal.jl")
include("test_BlockJacobi.jl")
include("test_SPAI.jl")
include("test_ProxyGmres.jl")
include("test_ILU.jl")
include("test_Jacobi.jl")
include("test_SOR.jl")
