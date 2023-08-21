"""
    SPAI{T, S<:AbstractVector{T}} <: Preconditioner

    # Fields 
    `M::S` - The sparse Matrix preconditioner.

"""

mutable struct SPAI{T, S<:AbstractMatrix{T}} <: Preconditioner
    M::S
end

Base.eltype(::SPAI{T, S}) where {T, S} = T

"""
    DiagonalPreconditioner(A::AbstractMatrix)
    
    # Arguments
    `A::AbstractMatrix` - The matrix being preconditioned.
    `Z::Union{SparseMatrixCSC, BandedMatrix}` - The sparsity pattern to be used for the approximate inversion.

    # Returns
    `SPAI{T, S}` - Returns SPAI datatype which contains a matrix matching the sparsity pattern of Z.

    #References 
    Grote, M. J., & Huckle, T. (1997). Parallel preconditioning with sparse 
    approximate inverses. SIAM Journal on Scientific Computing, 18(3), 838-853.
"""

function SPAI(A::AbstractMatrix, Z::Union{SparseMatrixCSC, BandedMatrix})
    m = size(A, 1)
    Ta = eltype(A)
    Tz = eltype(Z)
    Id = zeros(Ta, m)
    @assert Tz <: Ta "Make sure that your nonzero pattern Matrix has the same element type as the matrix you are preconditioning."
    if typeof(Z) <: BandedMatrix
        (u, l) = bandwidths(Z)
        for i in 1:m
            Id[i] = 1
            #Determine the nonzero pattern in this column
            if i > u && i < m-l
                rdx = max(i-u, 1):min(i+l, m)
            elseif i <= u
                rdx = 1:min(i+l, m)
            else
                rdx = max(i-u, 1):m
            end

            @views Z[rdx, i] .= qr(A[:, rdx]) \ Id #Fill in least squares solution
            Id[i] = 0 #Reset constant vector
        end
    else 
        nzr, nzc, _ = findnz(Z)
        rdx = Array{Int}(undef, m)
        low = 1
        n = length(nzc)
        for i in 1:m
            Id[i] = 1
            k,low = SparseRdx!(rdx, nzr, nzc, i, n, low = low) #Determine nonzero pattern in this column
            rdv = view(rdx, 1:k)
            @views Z[rdv, i] .= qr(A[:, rdv]) \ Id #Fill in least squares solution.
            Id[i] = 0
        end
    end

    return SPAI{Ta, typeof(A)}(Z)
end

#Functions that apply preconditioner (all are equivalent)
function mul!(x::AbstractVector, P::SPAI, y::AbstractVector)
    mul!(x, P.M, y)
end

function mul!(P::SPAI, y::AbstractVector)
    x = deepcopy(y)
    mul!(y, P.M, x)
end

function ldiv!(x::AbstractVector, P::SPAI, y::AbstractVector)
    mul!(x, P.M, y)
end

function ldiv!(P::SPAI, y::AbstractVector)
    x = deepcopy(y)
    mul!(y, P.M, x)
end

function (\)(P::SPAI, y::AbstractVector)
    x = deepcopy(y)
    mul!(x, P.M, y)
    return x
end

#Function to search fpr appropiate indicies in a way that limits passes through vector of nonzeros
function SparseRdx!(rdx::Array{Int}, nzr::Array{Int}, nzc::Array{Int}, j::Int, n::Int; low = 1)
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

