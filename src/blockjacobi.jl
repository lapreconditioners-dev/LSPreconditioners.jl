"""
    BlockJacobi{T, F<:Factorization{T}} <: Preconditioner

    # Fields 
    `nblocks::Int` - Number of blocks in the preconditioner.
    `blocksizes::Array{Int}` - Array containing the blocks sizes for each preconditioner.
    `blocks::Vector{F}` - Array containing the factorizations of the various blocks.
"""
mutable struct BlockJacobi{T, F<:Factorization{T}} <: Preconditioner
    nblocks::Int
    blocksizes::Array{Int}
    blocks::Vector{F}
end

Base.eltype(::BlockJacobi{T,F}) where {T, F} = T

"""
    DiagonalPreconditioner(A::AbstractMatrix)
    
    # Arguments
   `A::AbstractMatrix` - The matrix being preconditioned.
    `blocksize::Int` - The desired size of the blocks. This will be true for all blocks except the final one. 

    # Returns
    `BlockJacobi{T, S}` - Returns BlockJacobi datatype which contains the number of blocks, block sizes, and factorizations.
"""
function BlockJacobi(A::AbstractMatrix, blocksize::Int)
    m = size(A, 1)
    remB = rem(m, blocksize)
    nblocks = div(m, blocksize) + (remB == 0 ? 0 : 1)
    bsizes = Array{Int}(undef, nblocks)
    T = eltype(A)
    for i in 1:nblocks-1
        bsizes[i] = blocksize
    end

    bsizes[nblocks] = remB == 0 ? blocksize : remB
    blocks = get_blocks(A, nblocks)
    endp = 0
    for i in 1:nblocks
        startp = endp + 1
        endp = startp + bsizes[i] - 1
        blocks[i] = lu(A[startp:endp, startp:endp])
    end

    return BlockJacobi{eltype(A), typeof(blocks[1])}(nblocks, bsizes, blocks)
end

function BlockJacobi(A)
    blocksize = min(Int(ceil(size(A, 1)/10)), 100)
    return BlockJacobi(A, blocksize)
end

#Function to apply block jacobi preconditioner allow one and two entry versions
function mul!(x::AbstractVector, P::BlockJacobi, y::AbstractVector)
    endp = 0
    for i in 1:P.nblocks
        startp = endp + 1 
        endp = startp + P.blocksizes[i] - 1 
        @views ldiv!(x[startp:endp], P.blocks[i], y[startp:endp])
    end
end

function mul!(P::BlockJacobi, y::AbstractVector)
    endp = 0
    for i in 1:P.nblocks
        startp = endp + 1 
        endp = startp + P.blocksizes[i] - 1 
        @views ldiv!(P.blocks[i], y[startp:endp])
    end
end

function ldiv!(x::AbstractVector, P::BlockJacobi, y::AbstractVector)
    endp = 0
    for i in 1:P.nblocks
        startp = endp + 1 
        endp = startp + P.blocksizes[i] - 1 
        @views ldiv!(x[startp:endp], P.blocks[i], y[startp:endp])
    end
end

function ldiv!(P::BlockJacobi, y::AbstractVector)
    endp = 0
    for i in 1:P.nblocks
        startp = endp + 1 
        endp = startp + P.blocksizes[i] - 1 
        @views ldiv!(P.blocks[i], y[startp:endp])
    end
end

function (\)(P::BlockJacobi, y::AbstractVector)
    x = deepcopy(y)
    ldiv!(P, x)
    return x
end

#Set of functions to return vector of blocks with correct LU decomposition type
function get_blocks(A::SparseMatrixCSC, nblocks)
    T = eltype(A)
    return Vector{UmfpackLU{T, Int}}(undef, nblocks)
end 

function get_blocks(A::Matrix, nblocks)
    T = eltype(A)
        return Vector{LU{T, Matrix{T}, Vector{LinearAlgebra.BlasInt}}}(undef, nblocks)
end 

function get_blocks(A::BandedMatrix, nblocks)
    T = eltype(A)
    return Vector{BandedLU{T, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}}}(undef, nblocks) 
end 
