#Simple Implementations of the applications of the ProxyGMRES functions 
#Function for full polynomial application
function ApplyPol(v,A,H,a)
    m = length(v)
    n = size(H,2)
    V = zeros(ComplexF64,m,n)
    @views copyto!(V[:,1],v)
    V[:,1]./= sqrt(n)
    for i = 1:n-1
        @views V[:,i+1] = 1/H[i+1,i] * (A * V[:,i] - V[:,1:i] * H[1:i,i]) 
    end

    if eltype(A) <: Complex
        return V * a
    else
        return real(V * a)
    end

end

#Function for application of short recurrence polynomail
function ApplyPolSR(v,A,H,a,k)
    m = length(v)
    n = size(H,2)
    V = zeros(ComplexF64,m,n)
    @views copyto!(V[:,1],v)
    V[:,1]./= sqrt(n)
    for i = 1:n-1
        if i > k
            @views V[:,i+1] = 1/H[i+1,i] * (A * V[:,i] - V[:,(i-k+1):i] * H[(i-k+1):i,i])
        else 
            @views V[:,i+1] = 1/H[i+1,i] * (A * V[:,i] - V[:,1:i] * H[1:i,i])
        end

    end

    if eltype(A) <: Complex
        return V * a
    else
        return real(V * a)
    end
end

#Funtion for application of a compound polynomial
function ApplyCompound(v,A,H1,a1,k1,H2,a2,k2)
    m = length(v)
    n = size(H2,2)    
    V = zeros(ComplexF64,m,n)
    @views copyto!(V[:,1],v)
    V[:,1]./= sqrt(n)
    for i = 1:n-1
        if i > k2
            @views V[:,i+1] = 1/H2[i+1,i] * (A * ApplyPolSR(V[:,i],A,H1,a1,k1) - V[:,(i-k2+1):i] * H2[(i-k2+1):i,i]) 
        else 
            @views V[:,i+1] = 1/H2[i+1,i] * (A * ApplyPolSR(V[:,i],A,H1,a1,k1) - V[:,1:i] * H2[1:i,i])
        end
        
    end

    if eltype(A) <: Complex
        b = V * a2
    else
        b = real(V * a2)
    end
    return ApplyPolSR(b,A,H1,a1,k1)
end

@testset "ProxyGmresApply" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Test for the mul! function
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        P = ProxyGmres(A, 10, 10)
        mul!(x, P, b)
        @test x ≈ ApplyPol(b, A, P.H, P.a) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        P = ProxyGmres(A, 10, 10)
        mul!(x, P, b)
        @test x ≈ ApplyPol(b, A, P.H, P.a) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        P = ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPol(b, A, P.H, P.a) 
        
        #Test for the ldiv! function
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = ProxyGmres(A, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        @test P \ b ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        P = ProxyGmres(A, 10, 10)
        ldiv!(x, P, b)
        @test x ≈ ApplyPol(b,A,P.H,P.a) 
        @test P \ b ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = ProxyGmres(A, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        @test P \ b ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        P = ProxyGmres(A, 10, 10)
        ldiv!(x, P, b)
        @test x ≈ ApplyPol(b, A, P.H, P.a) 
        @test P \ b ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = ProxyGmres(A, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        @test P \ b ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        P = ProxyGmres(A, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyPol(b, A, P.H, P.a) 
        @test P \ b ≈ ApplyPolSR(b, A, P.H, P.a, P.k) 
        
        #Test for the one entry mul! function
        b = rand(FC, 10)
        c = deepcopy(b)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = ProxyGmres(A, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyPolSR(c, A, P.H, P.a, P.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 10)
        mul!(P, b)
        @test b ≈ ApplyPol(c, A, P.H, P.a) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyPolSR(c,A,P.H,P.a,P.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 10)
        mul!(P, b)
        @test b ≈ ApplyPol(c, A, P.H, P.a) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyPolSR(c, A, P.H, P.a, P.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyPol(c, A, P.H, P.a) 
        
        #Test for the one entry ldiv! function
        b = rand(FC, 10)
        c = deepcopy(b)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = ProxyGmres(A, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyPolSR(c, A, P.H, P.a, P.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 10)
        ldiv!(P, b)
        @test b ≈ ApplyPol(c, A, P.H, P.a) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyPolSR(c,A,P.H,P.a,P.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 10)
        ldiv!(P, b)
        @test b ≈ ApplyPol(c, A, P.H, P.a) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyPolSR(c, A, P.H, P.a, P.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = ProxyGmres(A, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyPol(c, A, P.H, P.a) 
    end
end

@testset "CompoundProxyGmresApply" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Test the mul! function
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H ,P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        
        #Test the ldiv! function
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        ldiv!(x, P, b)
        @test x ≈ ApplyCompound(b, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 

        #Test the one entry mul! function
        b = rand(FC, 10)
        c = deepcopy(b)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        
        #Test the one entry ldiv! function
        b = rand(FC, 10)
        c = deepcopy(b)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        ldiv!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        ldiv!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 3, 2, 2)
        ldiv!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
        b = rand(FC, 10)
        c = deepcopy(b)
        P = CompoundProxyGmres(A, 10, 2, 10, 2)
        ldiv!(P, b)
        @test b ≈ ApplyCompound(c, A, P.P1.H, P.P1.a, P.P1.k, P.P2.H, P.P2.a, P.P2.k) 
    end
end

