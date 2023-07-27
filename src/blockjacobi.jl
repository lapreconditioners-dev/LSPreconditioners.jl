mutable struct BlockJacobi{T, S<:AbstractMatrix{T}} <: LSPreconditioners.Preconditioner
    nblocks::Int64
    blocksizes::Array{Int}
    blocks::Vector{Union{LU{T, Matrix{T}, Vector{Int}}, UmfpackLU{T, Int}}}
end

Base.eltype(::BlockJacobi{T, S}) where {T, S} = T

# Function to form a block Jacobi preconditioner
function BlockJacobi(A::AbstractMatrix, blocksize::Integer)
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

    return BlockJacobi{eltype(A), typeof(A)}(nblocks, bsizes, blocks)
end

function BlockJacobi(A)
    blocksize = min(Int(ceil(size(A, 1)/10)), 100)
    return BlockJacobi(A, blocksize)
end

#Function to apply block jacobi preconditioner
function LinearAlgebra.mul!(x, P::BlockJacobi, y)
    endp = 0
    for i in 1:P.nblocks
        startp = endp + 1 
        endp = startp + P.blocksizes[i] - 1 
        ldiv!(x[startp:endp], P.blocks[i], y[startp:endp])
    end
end

function get_blocks(A::SparseMatrixCSC, nblocks)
    T = eltype(A)
    return Vector{UmfpackLU{T, Int}}(undef, nblocks)
end 

function get_blocks(A::Matrix, nblocks)
    T = eltype(A)
    return Vector{LU{T, Matrix{T}, Vector{Int}}}(undef, nblocks)
end 

function get_blocks(A::BandedMatrix, nblocks)
    T = eltype(A)
    return Vector{BandedMatrices.BandedLU{T, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}}}(undef, nblocks) 
end 
