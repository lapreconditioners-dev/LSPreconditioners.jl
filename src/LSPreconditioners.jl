module LSPreconditioners

import LinearAlgebra: mul!, diag
abstract type Preconditioner end

include("diagonal.jl")

export DiagonalPreconditioner

end # module LSPreconditioners
