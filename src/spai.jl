using BandedMatrices
using SparseArrays

#Implementation of SPAI preconditoner which allows one to specify a sparsity pattern stored in a matrix Z and it formulates a preconditioner that fits into the sparisty pattern

mutable struct SPAI{T, S<:AbstractMatrix{T}} <: LSPreconditioners.Preconditioner
    M::S
end

Base.eltype(::SPAI{T, S}) where {T, S} = T

function SPAI(A::AbstractMatrix, Z::Union{SparseMatrixCSC, BandedMatrix})
    m = size(A, 1)
    T = eltype(A)
    Id = zeros(m)
    if typeof(Z) <: BandedMatrix
        (u, l) = bandwidths(Z)
        for i in 1:m
            Id[i] = 1
            if i > u && i < m-l
                rdx = i-u:i+l
            elseif i <= u
                rdx = 1:i+l
            else
                rdx = i-u:m
            end

            @views Z[rdx, i] .= qr(A[:, rdx]) \ Id
            Id[i] = 0
        end
    else 
        nzr, nzc, _ = findnz(Z)
        for i in 1:m
            Id[i] = 1
            rdx = nzr[nzc .== i]
            @views Z[rdx, i] .= qr(A[:, rdx]) \ Id
            Id[i] = 0
        end
    end

    return SPAI{eltype(A), typeof(A)}(Z)
end

function LinearAlgebra.mul!(x, P::SPAI, y)
    mul!(x, P.M, y)
end
