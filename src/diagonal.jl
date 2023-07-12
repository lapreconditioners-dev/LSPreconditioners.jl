mutable struct DiagonalPreconditioner{T, S<:AbstractVector{T}} <: Preconditioner
    D::S
    y::S
end

function DiagonalPreconditioner(A::AbstractMatrix)
    D = Array(diag(A, 0))
    y = similar(D)
    return DiagonalPreconditioner{eltype(D), typeof(D)}(D, y)
end

Base.eltype(::DiagonalPreconditioner{T, S}) where {T, S} = T

@inline function mul!(y::AbstractVector, C::DiagonalPreconditioner, b::AbstractVector)
    @inbounds @simd for i in 1:length(C.D)
        y[i] = b[i] * (1 / C.D[i])
    end
end
