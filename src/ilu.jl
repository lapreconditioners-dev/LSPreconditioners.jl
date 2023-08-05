"""
    This is a set of wrapper functions that combines the functionality of IncompleteLU.jl and ILUZero.jl.
"""

"""
    ILU{T, S} <: LSPreconditioner

    # Fields 
    F::Union{ILU0Precon, ILUFactorization} - Contains the two types of ILU decompositions.
"""
mutable struct ILU{T, S} <: LSPreconditioners.Preconditioner
    F::Union{ILU0Precon, ILUFactorization}
end

Base.eltype(::ILU{T, S}) where {T, S} = T

"""
    ILU(A::SparseMatrixCSC, t)
    
    # Arguments
    A::SparseMatrixCSC - The matrix being preconditioned.
    t::Float64 - The threshold at which LU decomposition valuess are set to zero.

    # Returns
    ILU::ILU - Returns datatype of ILU which contains an ILUFactorization from IncompleteLU.jl.
"""
function ILU(A::SparseMatrixCSC; t = 1e-3)
    T = eltype(A)
    return ILU{T, Int}(ilu(A, τ=t))
end

"""
    ILU(A::SparseMatrixCSC)
    
    # Arguments
    A::SparseMatrixCSC - The matrix being preconditioned.

    # Returns
    ILU::ILU - Returns ILU datatype which contains a ILU0Precon, which is found in ILUZero.jl.
"""
function ILU(A::SparseMatrixCSC)
    T = eltype(A)
    return ILU{T, Int}(ilu0(A))
end

function LinearAlgebra.mul!(x::AbstractVector, Decomp::ILU, y::AbstractVector)
    ldiv!(x, Decomp.F, y)
end
