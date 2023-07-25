module LSPreconditioners

import LinearAlgebra: mul!, diag, LU
abstract type Preconditioner end

include("diagonal.jl")
include("blockJacobi.jl")

export DiagonalPreconditioner
export BlockJacobi

preconditioner_types = [DiagonalPreconditioner, BlockJacobi]

end # module LSPreconditioners
