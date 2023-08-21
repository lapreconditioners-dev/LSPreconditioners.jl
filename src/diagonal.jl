"""
    DiagonalPreconditioner{T, S<:AbstractVector{T}} <: Preconditioner

    # Fields 
    `D::S` - Contains the inverse of the diagonal of the matrix being preconditioned.
    `y::S` - A buffer vector.
"""
mutable struct DiagonalPreconditioner{T, S<:AbstractVector{T}} <: Preconditioner
    D::S
    y::S
end

"""
    DiagonalPreconditioner(A::AbstractMatrix)
    
    # Arguments
    `A::AbstractMatrix` - The matrix being preconditioned.

    # Returns
    `DiagonalPreconditioner{T}` - Returns DiagonalPreconditioner datatype which contains a buffer vector and the inverted diagonal.
"""
function DiagonalPreconditioner(A::AbstractMatrix)
    nonzero = diag(A,0) .== 0
    if sum(nonzero) < 1
        D = Array(1 ./ diag(A, 0)) #division is more expensive then multiplication so pre-divide
    else
        @warn "At least one diagonal entry is zero proceeding by ignoring that entry."
        temp = diag(A,0)
        temp[nonzero] .= 1
        D = Array(1 ./ temp)
    end
    y = similar(D)
    return DiagonalPreconditioner{eltype(D), typeof(D)}(D, y)
end

Base.eltype(::DiagonalPreconditioner{T, S}) where {T, S} = T

#Functions to apply the preconditioner
@inline function mul!(x::AbstractVector, C::DiagonalPreconditioner, y::AbstractVector)
    ln = length(C.D)
    @inbounds @simd for i in 1:ln
        x[i] = y[i] * C.D[i]
    end
end

@inline function mul!(C::DiagonalPreconditioner, y::AbstractVector)
    ln = length(C.D)
    @inbounds @simd for i in 1:ln
        y[i] *= C.D[i]
    end
end

@inline function ldiv!(x::AbstractVector, C::DiagonalPreconditioner, y::AbstractVector)
    ln = length(C.D)
    @inbounds @simd for i in 1:ln
        x[i] = y[i] * C.D[i]
    end
end

@inline function ldiv!(C::DiagonalPreconditioner, y::AbstractVector)
    ln = length(C.D)
    @inbounds @simd for i in 1:ln
        y[i] *= C.D[i]
    end
end

function (\)(C::DiagonalPreconditioner, y::AbstractVector)
    x = deepcopy(y)
    ldiv!(C, x)
    return x
end
