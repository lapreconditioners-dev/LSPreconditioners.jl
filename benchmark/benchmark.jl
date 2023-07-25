using LinearAlgebra
using SparseArrays
using LSPreconditioners
using Krylov
using MatrixDepot
using Random
Random.seed!(1234)
itmax = 100

coefficients = 1:6
ntests = length(coefficients)
iterations = zeros(Int64, ntests, 3)

for i in 3:ntests
    n = 2^coefficients[i]
    A = matrixdepot("randsvd", n, 1e2)
    b = rand(n)
    dump(b)
    (x, stats) = bicgstab(A, b;itmax=itmax, history=true)
    iterations[i, 1] = stats.niter
    
    PA = DiagonalPreconditioner(A)
    (x, stats) = bicgstab(A, b; M=PA, itmax=itmax, history=true)
    iterations[i, 2] = stats.niter
end

