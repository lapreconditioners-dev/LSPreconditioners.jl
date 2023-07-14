module LSPreconditioners

import LinearAlgebra: mul!, diag
abstract type Preconditioner end

include("diagonal.jl")

export DiagonalPreconditioner

include("blockJacobi.jl")

export BlockJacobi

end # module LSPreconditioners
