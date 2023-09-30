"""
    Jacobi{T, S<:AbstractVector{T}, M<:AbstractVector{T}} <: Preconditioner

    # Fields
    `n::Int` -  The dimension of the matrix.
    `maxit::Int` - The maximum number of iterations that Jacobi will be run.
    `thres::Union{Float64, Nothing}` - The threshold for the convergence criteria of the residuals, can either be nothing in which iteration stops when the maximum iteration is achieved, or some floating point value in which case the residual is checked.
    `A::M` - Contains the original matrix.
    `buff::S` - A buffer vector.
    `res::S` - A vector to store the residual.
"""
mutable struct Jacobi{T, S<:AbstractVector{T}, M<:AbstractMatrix{T}} <: LSPreconditioners.Preconditioner
    n::Int
    maxit::Int
    thres::Union{Float64, Nothing}
    A::M
    D::S
    buff::S
    res::S
end

"""
    Jacobi(A::AbstractMatrix; maxit=size(A,1), thres=1e-10)
    
    # Arguments
    `A::AbstractMatrix` - The matrix being preconditioned.
    `maxit::Int` - The maximum number of iterations for the Jacobi solver.
    `thres::Float64` - The convergence criteria for the residual of the Jacobi solver. 

    # Returns
    `Jacobi{T}` - Returns Jacobi datatype.
"""
function Jacobi(A::AbstractMatrix; maxit=size(A,1), thres=nothing)
    T = eltype(A)
    n = size(A, 1)
    buff = Array{T}(undef, n)
    res = Array{T}(undef, n)
    D = diag(A)
    if sum(D .== 0) > 0
        return error("Your matrix has a zero diagonal, remedy this and try again")
    end
    return Jacobi{T, typeof(buff), typeof(A)}(n, maxit, thres, A, D, buff, res)
end

Base.eltype(::Jacobi{T, S, M}) where {T, S, M} = T

function mul!(P::Jacobi{T, S, M}, y::AbstractVector{T}) where {T, S, M}
    Ty = eltype(P)
    x = zeros(Ty, P.n) 
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

    copy!(y, x) 
end

function ldiv!(P::Jacobi, y::AbstractVector)
    Ty = eltype(P)
    x = zeros(Ty, P.n) 
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

    copy!(y, x) 
end

function (\)(P::Jacobi, y::AbstractVector)
    Ty = eltype(P)
    x = zeros(Ty, P.n) 
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

    return x 
end

function mul!(x::AbstractVector, P::Jacobi, y::AbstractVector)
    fill!(x, 0)
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
        end

    else
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

end

function ldiv!(x::AbstractVector, P::Jacobi, y::AbstractVector)
    fill!(x, 0)
    if typeof(P.thres) <: Nothing #Check if threshold will be used
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
        end
        
    else
        for k in 1:P.maxit
            iterate_jacobi!(x, P, y)
            copy!(P.res, y)
            mul!(P.res, P.A, x, -1, 1) #Compute the residual 
            if norm(P.res) < P.thres
                break
            end

        end

    end

end

@inline function iterate_jacobi!(x::AbstractVector{T}, P::Jacobi{T, S, M}, y::AbstractVector{T}) where {T, S, M} 
    fill!(P.buff, 0)
    for j in 1:P.n
        @simd for i in 1:(j-1)
            @inbounds P.buff[i] += P.A[i, j] * x[j] #Loop through the rows since col orient
	    end

        @simd for i in (j+1):P.n
           @inbounds P.buff[i] += P.A[i, j] * x[j]
	    end

	end
    x .= (y .- P.buff) ./ P.D

end
