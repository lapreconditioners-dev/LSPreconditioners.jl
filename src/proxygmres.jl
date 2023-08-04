"""
    ProxyGmres{T}
    Array wrapper for proxy gmres function

    # Fields
    - `A::AbstractMatrix{T}`: The matrix being preconditioned
    - `V::Matrix{ComplexF64}`: The buffer matrix to store the recursive vectors.
    - `a::Vector{ComplexF64}`: The coefficients from the polynomial coefficients.
    - `H::Matrix{ComplexF64}`: The upper Hessenberg matrix from polynomial gmres.
    - `vs::Vector{ComplexF64}`: Storage vector for matrix vector multiplications.
    - `hs::Vector{ComplexF64}`: The reorder buffer vector for recrsive operations.
    - `vc::Vector{ComplexF64}`: Storage buffer for solution.
    - `k::Int`:The length of the partial orthoogonalization.

    # References:
    * Xin Ye, Yuanzhe Xi, and Yousef Saad. 2021. Proxy-GMRES: Preconditioning via GMRES in Polynomial Space. SIAM J. Matrix Anal. Appl. 42, 3 (January 2021), 1248–1267. https://doi.org/10.1137/20M1342562
"""
mutable struct ProxyGmres{T} <: LSPreconditioners.Preconditioner
    A::AbstractMatrix{T}
    V::Matrix{ComplexF64}
    a::Vector{ComplexF64}
    H::Matrix{ComplexF64}
    vs::Vector{ComplexF64}
    hs::Vector{ComplexF64}
    vc::Vector{ComplexF64}
    k::Int
end

"""
    ProxyGmres{T}
    Array wrapper for proxy gmres function
    
    # Fields
    - `P1::ProxyGmres`: Inner polynomial preconditioner in compound sequence.
    - `P2::ProxyGmres`: Outer polynomial preconditioner in compound sequence.
"""
mutable struct CompoundProxyGmres{T} <: LSPreconditioners.Preconditioner
    P1::ProxyGmres
    P2::ProxyGmres
end

"""
    ProxyGmres(A::AbstractMatrix, d::Int, k::Int)

    # Arguments
    - `A::AbstractMatrix`: The matrix beinng preconditioned.
    - `d::Int`: The degree of the polynomial.
    - `k::Int`: The width of the partial orthogonalization.
    # Returns
    - `ProxyGmres{T}`: The preconditioner information.
"""
function ProxyGmres(A::AbstractMatrix, d::Int, k::Int)
    m,n = size(A)
    T = eltype(A)
    ai = min(div(m,2), 100) #If matrix is small enough compute all ritz vectors otherwise just 100
    H,_ = Arnoldi(A, ones(n), ai)
    evs = eigen(H[1:ai, 1:ai]).values
    if sum(imag(evs) .!= 0) > 0 #check that eigenvalues are imaginary
        z = genBound(evs, 2 * d)
    else
        delt = 1 / (2 * d - 1)
        z = minimum(real.(evs)):delt:maximum(real.(evs))
    end

    a,H,k = ComputePoly(z, d, k)
    V = zeros(ComplexF64, m, k)
    vc =  zeros(ComplexF64, m)
    vs = zeros(ComplexF64, m)
    hs = zeros(ComplexF64, k)
    return ProxyGmres{T}(A, V, a, H, vs, hs, vc, k)
end

"""
    CompoundProxyGmres(A::AbstractMatrix, d1::Int, d2::Int, k1::int, k2::int; radius = 1)
    
    # Arguments
    - `A::AbstractMatrix`: The matrix beinng preconditioned.
    - `d1::Int`: The degree of the inner polynomial.
    - `d2::Int`: The degree of the outer polynomial.
    - `k1::Int`: The width of the inner polynomial for the partial orthogonalization.
    - `k2::Int`: The width of the outer polynomial for the partial orthogonalization.

    # Keywords
    - `radius::Float`: The radius of the circle boundary aligning with the outer polynomial.

    # returns
    - `CompoundProxyGmres{T}`: The preconditioner information.
"""
function CompoundProxyGmres(A::AbstractMatrix, d1::Int, d2::Int, k1::Int, k2::Int; radius = 1)
    m,n = size(A)
    T = eltype(A)
    ai = min(m, 100)
    H,_ = Arnoldi(A, rand(n), ai)
    evs = eigen(H[1:ai, 1:ai]).values
    V1 = zeros(ComplexF64, m, k1)
    V2 = zeros(ComplexF64, m, k2+1)
    vc1 = zeros(ComplexF64, m)
    vc2 = zeros(ComplexF64, m)
    vs1 = zeros(ComplexF64, m)
    vs2 = zeros(ComplexF64, m)
    hs1 = zeros(ComplexF64, k1)
    hs2 = zeros(ComplexF64, k2)
    if sum(imag(evs) .!= 0) > 0
        z1 = genBound(evs, 2 * d1)
        z2 = GenCircle(1, radius, 2 * d2)
    else
        delt = 1 / (2 * d1 - 1)
        z1 = minimum(real.(evs)):delt:maximum(real.(evs))
        z2 = 0:delt:div(radius,1)
    end
    
    a1,H1,k1 = ComputePoly(z1, d1, k1)
    a2,H2,k2 = ComputePoly(z2, d2, k2)
    return CompoundProxyGmres{T}(
                                    ProxyGmres{T}(A, V1, a1, H1, vs1, hs1, vc1, k1), 
                                    ProxyGmres{T}(A, V2, a2, H2, vs2, hs2, vc2, k2)
                                )
end

function LinearAlgebra.mul!(x::AbstractVector, P::ProxyGmres, y::AbstractVector)
    copy!(x, y)
    ApplyPoly!(x, P)
end

function LinearAlgebra.mul!(x::AbstractVector, P::CompoundProxyGmres, y::AbstractVector)
    copy!(x, y)
    ApplyCompoundPoly!(x, P)
end

function ComputePoly(z::AbstractArray, m::Integer, k::Integer; tol = 1e-12)
    n = length(z)
    T = eltype(z)
    H = zeros(T, m+1, m)
    Q = Array{T,2}(undef, n, m+1)
    norm1 = sqrt(n)
    Q[:, 1] .= 1/norm1
    #Do polynomial arnoldi
    for i in 1:m
        Q[:, i+1] .= z .* @view(Q[:, i])
        for j in max(1, i-k+1):i
            H[j, i] = dot(@view(Q[:, i+1]), @view(Q[:, j]))
            Q[:, i+1] .-= H[j, i] * @view(Q[:, j])
        end

        nq = norm(Q[:, i+1])
        if nq < tol
            x = Array{T}(undef, i)
            if m > k #Check if full orthogonalization
                cons = ones(ComplexF64, i)
                x = (@view(Q[1:i, 1:i]) * @view(H[1:i, 1:i])) \ cons
            else
                cons = zeros(ComplexF64, i)
                cons[1] = norm1
                x = @view(H[1:i, 1:i]) \ cons
            end   

            return x[1:i], H[1:i, 1:i], k
        end

        H[i+1, i] = nq 
        Q[:, i+1] ./= nq 
    end

    if m > k #Check full orthogonalization
        cons = ones(ComplexF64, n)
        x = (Q*H) \ cons
    else
        cons = zeros(ComplexF64, m + 1)
        cons[1] = norm1
        x = H \ cons
    end   

    return x, H, k
end

function ApplyPoly!(v::AbstractVector, P::ProxyGmres)
    m = length(v)
    n = size(P.H, 2)
    k = P.k
    start = 0
    copyto!(P.vs, v)
    P.vs ./= sqrt(n)
    fill!(P.vc, 0)
    axpy!(P.a[1], P.vs, P.vc) 
    idx = 1
    P.V[:, idx] .= P.vs
    for i in 1:n-1
        idxm = idx
        idx = idx < k ? idx + 1 : 1
        start = max(1, i-k+1) 
        # Perform V[:,i+1] = 1/P.H[i+1, i] * (P.A * V[:,i] - V[:,start:i] * P.H[start:i,i]) 
        mul!(P.vs, P.A, @view(P.V[:, idxm]))
        if idxm < idx && i > k
            reorderH!( P, i, i-idxm+1, i, start)
            mul!(P.vs, P.V, P.hs, (-1.0+0*im)/P.H[i+1, i], (1.0+0*im)/P.H[i+1, i]) 
        else
            start = max(1, i-k+1)
            reorderH!( P, i, start, i, i)
            mul!(P.vs, P.V, P.hs, (-1.0+0*im)/P.H[i+1, i], (1.0+0*im)/P.H[i+1, i]) 
        end

        P.V[:, idx] .= P.vs
        axpy!(P.a[i+1], P.vs, P.vc)
    end

    fill!(P.hs, 0)
    fill!(P.V, 0)
    # Check if the original vector is complex
    if eltype(v) <: Complex
        copyto!(v, P.vc)
    else
        copyto!(v, real(P.vc))
    end

end

function ApplyCompoundPoly!(v::AbstractVector, P::CompoundProxyGmres)
    m = length(v)
    n = size(P.P2.H, 2)
    k = P.P2.k
    #P2.V has one extra vector for storage of the result of the polynomial application partition. 
    P2V = view(P.P2.V, :, 1:k)
    Plv = view(P.P2.V, :, k+1) 
    start = 0
    copyto!(P.P2.vs, v)
    P.P2.vs ./= sqrt(n)
    fill!(P.P2.vc, 0)
    axpy!(P.P2.a[1], P.P2.vs, P.P2.vc) 
    idx = 1
    P.P2.V[:, idx] .= P.P2.vs
    for i in 1:n-1
        idxm = idx
        idx = idx < k ? idx + 1 : 1
        start = max(1, i-k+1) 
        # Perform V[:,i+1] = 1/P.H[i+1, i] * (P.A * Poly1(V[:,i]) - V[:,start:i] * P.H[start:i,i]) 
        mul!(Plv, P.P1, P.P2.V[:, idxm])
        mul!(P.P2.vs, P.P2.A, Plv)
        if idxm < idx && i > k
            reorderH!( P.P2, i, start, i, i)
            mul!(P.P2.vs, P2V, P.P2.hs, (-1.0+0*im)/P.P2.H[i+1, i], (1.0+0*im)/P.P2.H[i+1, i]) 
        else
            start = max(1, i-k+1)
            reorderH!( P.P2, i, start, i, i)
            mul!(P.P2.vs, P2V, P.P2.hs, (-1.0+0*im)/P.P2.H[i+1, i], (1.0+0*im)/P.P2.H[i+1, i]) 
        end

        P.P2.V[:, idx] .= P.P2.vs
        axpy!(P.P2.a[i+1], P.P2.vs, P.P2.vc)
    end

    ApplyPoly!(P.P2.vc, P.P1)
    fill!(P.P1.hs, 0)
    fill!(P.P2.hs, 0)
    fill!(P.P1.V, 0)
    fill!(P.P2.V, 0)
    # Check if the original vector is complex
    if eltype(P.P2.A) <: Complex
        copyto!(v, P.P2.vc)
    else
        copyto!(v, real(P.P2.vc))
    end

end

#Fucntion to reorder the h vector to align with the ordering of V
function reorderH!(P::ProxyGmres, i::Int,  start1::Int, end1::Int,  start2::Int)
    m = length(start1:end1)
    k = P.k
    l = start1
    #Handle to the left current index
    @inbounds @simd for j in 1:m
        P.hs[j] = P.H[l,i]
        l += 1
    end

    #Handle ordering to right of current index
    if i > k
        l = start2
        @inbounds @simd for j in m+1:k
            P.hs[j] = P.H[l,i]
            l += 1
        end

    end

end

#Matrix Arnoldi function for computation of a subset of the eigenvalues
function Arnoldi(A::AbstractMatrix, b::AbstractVector, n::Int)
    m,_ = size(A)
    Q = zeros(ComplexF64, m, m+1)
    H = zeros(ComplexF64, m+1, m)
    Q[:, 1] .= b
    nq = norm(Q[:, 1])
    Q[:, 1] ./= nq
    for i in 1:n
        @views mul!(Q[:, i+1], A, Q[:, i])
        for j in 1:i
            @views H[j, i] = dot(Q[:, i+1], Q[:, j])
            @views axpy!(-H[j, i], Q[:, j], Q[:, i+1])
        end

        H[i+1, i] = norm(Q[:, i+1])
        Q[:, i+1] ./= H[i+1, i]
    end

    return H, Q
end

#Converts complex number into a coordinate representation
function c2r(a::Complex)
    x = real(a)
    y = imag(a)
    return [x,y]
end

# Function that generates a specified number of intermediate points between two endpoints
function inter_points(p1, p2, npoints::AbstractVector)
    delt = 1 / (npoints - 1)
    k = 0.
    coordinates = Array{ComplexF64}(undef, npoints)
    for j in 1:npoints
        r = p1[1] * k + (1 - k) * p2[1]
        i =  p1[2] * k + (1 - k) * p2[2]
        coordinates[j] = r + i * im
        k += delt
    end

    return coordinates
end

#Generates a convex boundary around a set of points
function genBound(evs::AbstractVector, npoints::Int)
    coords = c2r.(evs)
    Hull = convex_hull(coords) #Create the hull around eigen values (Non Convex would be better)
    n = length(Hull)
    npoints_per = npoints
    if npoints_per == 0 
        print("Size of hull larger than number of points")
    end

    boundary = Array{ComplexF64}(undef, npoints_per * n) 
    for i in 1:n-1 #Between each vertex fill in the requisite number of points
       @views boundary[(i-1)*npoints_per+1:i*npoints_per] .= inter_points(Hull[i], Hull[i+1], npoints_per)
    end

    @views boundary[(n-1)*npoints_per+1:end] = inter_points(Hull[n], Hull[1], npoints_per)
    return boundary
end

#function to generate the boundary of a circle 
function GenCircle(center::Number, radius::Float, npoints::Float)
    tpoints = div(npoints,4)
    circle = Array{ComplexF64}(undef, tpoints*4)
    delt = radius * 1 / (tpoints - 1) 
    k = 0
    for i in 1:tpoints
        x = k
        diff = radius^2 - x^2 
        yp = sqrt(diff > 0 ? diff : 0) 
        yn = -yp
        #Compute coordinates in each quadrant of circle
        circle[i] = x + center + yp * im
        circle[tpoints+i] = x + center + yn * im
        circle[2*tpoints+i] = -x + center + yp * im
        circle[3*tpoints+i] = -x + center + yn * im
        k += delt
    end

    return circle
end
