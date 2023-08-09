module LSPreconditioners

import BandedMatrices: BandedMatrix, BandedLU, bandwidths
import ILUZero: ILU0Precon, ilu0
import IncompleteLU: ILUFactorization, ilu
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
#include("proxygmres.jl")
include("ilu.jl")

export DiagonalPreconditioner
export BlockJacobi
export SPAI
#export ProxyGmres, CompoundProxyGmres
export ILU

preconditioner_types = [DiagonalPreconditioner, BlockJacobi, SPAI, ILU]

end # module LSPreconditioners
