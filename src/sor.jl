"""
    SOR{T, S<:AbstractVector{T}, M<:AbstractVector{T}} <: Preconditioner

    # Fields
    `n::Int` -  The dimension of the matrix.
    `maxit::Int` - The maximum number of iterations that SOR will be run.
    `thres::Union{Float64, Nothing}` - The threshold for the convergence criteria of the residuals, can either be nothing in which iteration stops when the maximum iteration is achieved, or some floating point value in which case the residual is checked.
    `omega::T` - Contains the relaxation parameter.
    `A::M` - Contains the original matrix.
    `buff::S` - A buffer vector.
    `res::S` - A vector to store the residual.
"""
mutable struct SOR{T, S<:AbstractVector{T}, M<:AbstractMatrix{T}} <: LSPreconditioners.Preconditioner
    n::Int
    maxit::Int
    thres::Union{Float64, Nothing}
    omega::T
    A::M
    buff::S
    res::S
end

"""
    SOR(A::AbstractMatrix, omega::Float64; maxit=size(A,1), thres=1e-10)
    
    # Arguments
    `A::AbstractMatrix` - The matrix being preconditioned.
    `omega::Number` - The relaxation parameter.
    `maxit::Int` - The maximum number of iterations for the SOR solver.
    `thres::Float64` - The convergence criteria for the residual of the SOR solver. 

    # Returns
    `SOR{T}` - Returns SOR datatype.
"""
function SOR(A::AbstractMatrix, omega::Float64; maxit=size(A,1), thres=nothing)
    T = eltype(A)
    n = size(A, 1)
    buff = Array{T}(undef, n)
    res = Array{T}(undef, n)
    if omega < norm(T(1))
        @warn "omega should be at least 1"
    end
    return SOR{T, typeof(buff), typeof(A)}(n, maxit, thres, T(omega), A, buff, res)
end

Base.eltype(::SOR{T, S, M}) where {T, S, M} = T

function mul!(P::SOR{T, S, M}, y::AbstractVector{T}) where {T, S, M}
    Ty = eltype(P)
    fill!(P.buff, Ty(0)) 
    x = P.buff
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

    copy!(y, P.buff) 
end

function ldiv!(P::SOR, y::AbstractVector)
    Ty = eltype(P)
    fill!(P.buff, Ty(0)) 
    x = P.buff
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

    copy!(y, P.buff) 
end

function (\)(P::SOR, y::AbstractVector)
    Ty = eltype(P)
    fill!(P.buff, Ty(0)) 
    x = P.buff
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

    return P.buff
end
function mul!(x::AbstractVector, P::SOR, y::AbstractVector)
    fill!(x, 0)
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

end

function ldiv!(x::AbstractVector, P::SOR, y::AbstractVector)
    fill!(x, 0)
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
        end
        
    else
        for k in 1:P.maxit
            iterate_sor!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

end

@inline function iterate_sor!(x::AbstractVector{T}, P::SOR{T, S, M}, y::AbstractVector{T}) where {T, S, M} accum::T = zero(T)
    cons_diff::T = zero(T)
    for i in 1:P.n
        accum = zero(T) 
        @simd for j in 1:(i-1)
	        accum += P.A[i, j] * x[j]
	    end

        @simd for j in (i+1):P.n
	        accum += P.A[i, j] * x[j]
	    end

        cons_diff = y[i] - accum
        cons_diff /= P.A[i, i]
        x[i] += P.omega * (cons_diff - x[i])
	end

end
