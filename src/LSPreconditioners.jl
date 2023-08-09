module LSPreconditioners

import BandedMatrices: BandedMatrix, BandedLU, bandwidths 
import LazySets: convex_hull
import LinearAlgebra
import LinearAlgebra: mul!, diag, lu, LU, ldiv!, Factorization, qr
import SparseArrays: SparseMatrixCSC, findnz

using SuiteSparse: UMFPACK
using .UMFPACK: UmfpackLU

abstract type Preconditioner end

include("diagonal.jl")
include("blockJacobi.jl")
include("SPAI.jl")
include("proxygmres.jl")

export DiagonalPreconditioner
export BlockJacobi
export SPAI
export ProxyGmres, CompoundProxyGmres

preconditioner_types = [DiagonalPreconditioner, BlockJacobi, SPAI, ProxyGmres, CompoundProxyGmres]

end # module LSPreconditioners
