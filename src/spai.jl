using BandedMatrices
using SparseArrays

#Implementation of SPAI preconditoner which allows one to specify a sparsity pattern stored in a matrix Z and it formulates a preconditioner that fits into the sparisty pattern

mutable struct SPAI{T, S<:AbstractMatrix{T}} <: Preconditioner
    M::S
end
function SPAI(A::AbstractMatrix,Z::SparseMatrixCSC)
    m = size(A,1)
    T = eltype(A)
    nz = nnz(Z)
    v = Array{T}(undef,nz)
    nzr,nzc,_ = findnz(Z)
    M = sparse(nzr,nzc,v)
    Id = zeros(m)
    for i = 1:m
        Id[i] = 1
        rdx = nzr[nzc .== i]
        M[rdx,i] = A[:,rdx]\Id#ldiv!(M[rdx,i],qr(A[:,rdx]),Id)
        Id[i] = 0
    end
    return SPAI{eltype(A),typeof(A)}(M)
end

function SPAI(A::AbstractMatrix,Z::BandedMatrix)
    m = size(A,1)
    T = eltype(A)
    nz = nnz(Z)
    (u,l) = bandwidths(Z)
    M = BandedMatrix{T}(undef,(m,m),(u,l))
    Id = zeros(m)
    for i = 1:m
        Id[i] = 1
        if i>u && i<m-l
            rdx = i-u:i+l
        elseif i <= u
            rdx = 1:i+l
        else
            rdx = i-u:m
        end
         M[rdx,i] = A[:,rdx]\Id#ldiv!(M[rdx,i],qr(A[:,rdx]),Id)
        Id[i] = 0
    end
    return SPAI{eltype(A),typeof(A)}(M)
end

function LinearAlgebra.mul!(x,P::SPAI,y)
    mul!(x,P.M,y)
end
