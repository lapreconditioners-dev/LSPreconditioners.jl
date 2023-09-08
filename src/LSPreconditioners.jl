module LSPreconditioners

import BandedMatrices: BandedMatrix, BandedLU, bandwidths
import ILUZero: ILU0Precon, ilu0
import IncompleteLU: ILUFactorization, ilu
import LazySets: convex_hull
import LinearAlgebra
import LinearAlgebra: mul!, diag, lu, LU, ldiv!, Factorization, qr, norm, dot, axpy!, eigen, \
import SparseArrays: SparseMatrixCSC, findnz

using SuiteSparse: UMFPACK
using .UMFPACK: UmfpackLU

abstract type Preconditioner end

include("diagonal.jl")
include("blockjacobi.jl")
include("spai.jl")
include("proxygmres.jl")
include("ilu.jl")
include("sor.jl")

export DiagonalPreconditioner
export BlockJacobi
export SPAI
export ProxyGmres, CompoundProxyGmres
export ILU
export SOR

preconditioner_types = [BlockJacobi, CompoundProxyGmres, DiagonalPreconditioner, ILU, ProxyGmres, SPAI, SOR]

end # module LSPreconditioners
