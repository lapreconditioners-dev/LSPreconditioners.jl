using SuiteSparse
using SparseArrays
using LinearAlgebra

mutable struct BlockJacobi{T, S<:AbstractMatrix{T}} <: LSPreconditioners.Preconditioner
    nblocks::Int64
    blocksizes::Array{Int32}
    blocks::Vector{Union{LU{T, Matrix{T}, Vector{Int32}}, SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}}}
end

Base.eltype(::BlockJacobi{T, S}) where {T, S} = T

# Function to form a block Jacobi preconditioner
function BlockJacobi(A::Matrix, blocksize::Integer)
    m = size(A, 1)
    remB = rem(m, blocksize)
    nblocks = div(m, blocksize) + (remB == 0 ? 0 : 1)
    bsizes = Array{Int32}(undef, nblocks)
    T = eltype(A)
    for i in 1:nblocks-1
        bsizes[i] = blocksize
    end

    bsizes[nblocks] = remB == 0 ? blocksize : remB
    blocks = typeof(A) <: SparseMatrixCSC ? Vector{SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}}(undef, nblocks) : Vector{LU{T, Matrix{T}, Vector{Int32}}}(undef, nblocks)
    endp = 0
    for i in 1:nblocks
        startp = endp + 1
        endp = startp + bsizes[i] - 1
        @views blocks[i] = lu(A[startp:endp, startp:endp])
    end

    return BlockJacobi{eltype(A), typeof(A)}(nblocks, bsizes, blocks)
end

#Function to apply block jacobi preconditioner
function LinearAlgebra.mul!(x, P::BlockJacobi, y)
    endp = 0
    for i in 1:P.nblocks
        startp = endp + 1 
        endp = startp + P.blocksizes[i] - 1 
        @views ldiv!(x[startp:endp], P.blocks[i], y[startp:endp])
    end
end
