module LSPreconditioners

import LinearAlgebra
import LinearAlgebra: mul!, diag, lu, LU, ldiv!

import SparseArrays: SparseMatrixCSC
import BandedMatrices: BandedMatrix, BandedLU
using SuiteSparse: UMFPACK
using .UMFPACK: UmfpackLU

abstract type Preconditioner end

include("diagonal.jl")
include("blockJacobi.jl")

export DiagonalPreconditioner
export BlockJacobi

preconditioner_types = [DiagonalPreconditioner, BlockJacobi]

end # module LSPreconditioners
