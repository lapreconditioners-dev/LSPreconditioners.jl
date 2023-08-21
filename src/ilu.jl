"""
    This is a set of wrapper functions that combines the functionality of IncompleteLU.jl and ILUZero.jl.
"""

"""
    ILU{T, S} <: Preconditioner

    # Fields 
    `F::Union{ILU0Precon, ILUFactorization}` - Contains the two types of ILU decompositions.
"""
mutable struct ILU{T, S} <: Preconditioner
    F::Union{ILU0Precon, ILUFactorization}
end

Base.eltype(::ILU{T, S}) where {T, S} = T

"""
    ILU(A::SparseMatrixCSC)
    
    # Arguments
    `A::SparseMatrixCSC` - The matrix being preconditioned.

    # Returns
    `ILU{T,S}` - Returns ILU datatype which contains a ILU0Precon, which is found in ILUZero.jl.
"""
function ILU(A::SparseMatrixCSC)
    T = eltype(A)
    return ILU{T, Int}(ilu0(A))
end

"""
    ILU(A::SparseMatrixCSC, t)
    
    # Arguments
    `A::SparseMatrixCSC` - The matrix being preconditioned.
    `t::Float64` - The threshold at which LU decomposition valuess are set to zero.

    # Returns
    `ILU{T, S}` - Returns datatype of ILU which contains an ILUFactorization from IncompleteLU.jl.
"""
function ILU(A::SparseMatrixCSC, t = 1e-3)
    T = eltype(A)
    return ILU{T, Int}(ilu(A, τ=t))
end

#Functions to apply the preconditioner
function mul!(x::AbstractVector, Decomp::ILU, y::AbstractVector)
    ldiv!(x, Decomp.F, y)
end

function mul!(Decomp::ILU, y::AbstractVector)
    ldiv!(Decomp.F, y)
end

function ldiv!(x::AbstractVector, Decomp::ILU, y::AbstractVector)
    ldiv!(x, Decomp.F, y)
end

function ldiv!(Decomp::ILU, y::AbstractVector)
    ldiv!(Decomp.F, y)
end

function (\)(Decomp::ILU, y::AbstractVector)
    x = deepcopy(y)
    ldiv!(Decomp.F, x)
    return x
end
