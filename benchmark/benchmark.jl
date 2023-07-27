using LinearAlgebra
using SparseArrays
using LSPreconditioners
using Krylov
using MatrixDepot
using Random
Random.seed!(1234)
itmax = 100

coefficients = 1:3
ntests = length(coefficients)
iterations = zeros(Int64, ntests, length(LSPreconditioners.preconditioner_types) + 1)

ptypes = LSPreconditioners.preconditioner_types

for i in 1:ntests
    n = 2^coefficients[i]
    A = matrixdepot("wathen", n)
    b = rand(size(A, 1))
    
    (x, stats) = bicgstab(A, b; itmax=itmax, history=true)
    iterations[i, 1] = stats.niter

    for j=1:length(ptypes)
        PA = ptypes[j](A)
        (x, stats) = bicgstab(A, b; M=PA, itmax=itmax, history=true)
        iterations[i, j + 1] = stats.niter
    end
end

using Plots

plot(coefficients, iterations, label=["No preconditioner" ptypes...], xlabel="log2(n)", ylabel="Iterations", legend=:topleft)
