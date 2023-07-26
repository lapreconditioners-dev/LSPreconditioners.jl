module LSPreconditioners

import LinearAlgebra: mul!, diag, lu, LU
import SparseArrays: SparseMatrixCSC
import BandedMatrices
abstract type Preconditioner end

include("diagonal.jl")
include("blockJacobi.jl")

export DiagonalPreconditioner
export BlockJacobi

preconditioner_types = [DiagonalPreconditioner, BlockJacobi]

end # module LSPreconditioners
