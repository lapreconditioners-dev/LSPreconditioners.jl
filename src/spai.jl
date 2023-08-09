#Implementation of SPAI preconditoner which allows one to specify a sparsity pattern stored in a matrix Z and it formulates a preconditioner that fits into the sparisty pattern

mutable struct SPAI{T, S<:AbstractMatrix{T}} <: LSPreconditioners.Preconditioner
    M::S
end

Base.eltype(::SPAI{T, S}) where {T, S} = T

function SPAI(A::AbstractMatrix, Z::Union{SparseMatrixCSC, BandedMatrix})
    m = size(A, 1)
    Ta = eltype(A)
    Tz = eltype(Z)
    Id = zeros(Ta, m)
    if !(Tz <: Ta)
        return println("Make sure that your nonzero pattern is of the same type as your matrix")
    end
    if typeof(Z) <: BandedMatrix
        (u, l) = bandwidths(Z)
        for i in 1:m
            Id[i] = 1
            if i > u && i < m-l
                rdx = max(i-u,1):min(i+l,m)
            elseif i <= u
                rdx = 1:min(i+l,m)
            else
                rdx = max(i-u,1):m
            end

            @views Z[rdx, i] .= qr(A[:, rdx]) \ Id
            Id[i] = 0
        end
    else 
        nzr, nzc, _ = findnz(Z)
        rdx = Array{Integer}(undef,m)
        low = 1
        n = length(nzc)
        for i in 1:m
            Id[i] = 1
            k,low = SparseRdx!(rdx, nzr, nzc, i, n, low = low)
            rdv = view(rdx, 1:k)
            @views Z[rdv, i] .= qr(A[:, rdv]) \ Id
            Id[i] = 0
        end
    end

    return SPAI{Ta, typeof(A)}(Z)
end

function LinearAlgebra.mul!(x, P::SPAI, y)
    mul!(x, P.M, y)
end

#Function to search fpr appropiate indicies in a way that limits passes through vector of nonzeros
function SparseRdx!(rdx, nzr::Array{Int}, nzc::Array{Int}, j::Int, n::Int; low = 1)
    k = 0
    @inbounds for i in low:n
        if nzc[i] == j
            k += 1
            rdx[k] = nzr[i]
        else
            return k,i
        end

    end

    return k,n
end

