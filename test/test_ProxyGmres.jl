#=@testset "ProxyGmresApply" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPolSR(b,A,P.H,P.a,P.k) 
        P = LSPreconditioners.ProxyGmres(A, 10, 10)
        mul!(x, P, b)
        @test x ≈ ApplyPol(b,A,P.H,P.a) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPolSR(b,A,P.H,P.a,P.k) 
        P = LSPreconditioners.ProxyGmres(A, 10, 10)
        mul!(x, P, b)
        @test x ≈ ApplyPol(b,A,P.H,P.a) 
    end
    for FC in (Float64, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.ProxyGmres(A, 10, 2)
        mul!(x,P,b)
        @test x ≈ ApplyPolSR(b,A,P.H,P.a,P.k) 
        P = LSPreconditioners.ProxyGmres(A, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyPol(b,A,P.H,P.a) 
    end
end

@testset "CompoundProxyGmresApply" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b,A,P.P1.H,P.P1.a,P.P1.k,P.P2.H,P.P2.a,P.P2.k) 
        P = LSPreconditioners.CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b,A,P.P1.H,P.P1.a,P.P1.k,P.P2.H,P.P2.a,P.P2.k) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b,A,P.P1.H,P.P1.a,P.P1.k,P.P2.H,P.P2.a,P.P2.k) 
        P = LSPreconditioners.CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b,A,P.P1.H,P.P1.a,P.P1.k,P.P2.H,P.P2.a,P.P2.k) 
    end
    for FC in (Float64, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.CompoundProxyGmres(A, 10, 3, 2, 2)
        mul!(x,P,b)
        @test x ≈ ApplyCompound(b,A,P.P1.H,P.P1.a,P.P1.k,P.P2.H,P.P2.a,P.P2.k) 
        P = LSPreconditioners.CompoundProxyGmres(A, 10, 2, 10, 2)
        mul!(x, P, b)
        @test x ≈ ApplyCompound(b,A,P.P1.H,P.P1.a,P.P1.k,P.P2.H,P.P2.a,P.P2.k) 
    end
end
=#
#Simple Implementations of the applications of the ProxyGMRES functions 
function ApplyPol(v,A,H,a)
    m = length(v)
    n = size(H,2)
    V = zeros(ComplexF64,m,n)
    @views copyto!(V[:,1],v)
    V[:,1]./= sqrt(n)
    for i = 1:n-1
        @views V[:,i+1] = 1/H[i+1,i] * (A * V[:,i] - V[:,1:i] * H[1:i,i]) 
    end

    return real.(V * a)

end

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

    return real.(V * a)
end

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

    b = (V * a2)
    return real.(ApplyPolSR(b,A,H1,a1,k1))
end
