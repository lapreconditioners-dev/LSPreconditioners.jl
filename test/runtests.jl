using Test, LinearAlgebra, SparseArrays, BandedMatrices

println("Testing...")

include("test_diagonal.jl")
include("test_BlockJacobi.jl")
include("test_spai.jl")
