module LSPreconditioners

import LinearAlgebra
import LinearAlgebra: mul!, diag, lu, LU, ldiv!, Factorization, qr

import SparseArrays: SparseMatrixCSC, findnz
import BandedMatrices: BandedMatrix, BandedLU, bandwidths 
using SuiteSparse: UMFPACK
using .UMFPACK: UmfpackLU

abstract type Preconditioner end

include("diagonal.jl")
include("blockJacobi.jl")
include("SPAI.jl")

export DiagonalPreconditioner
export BlockJacobi
export SPAI

preconditioner_types = [DiagonalPreconditioner, BlockJacobi, SPAI]

end # module LSPreconditioners
