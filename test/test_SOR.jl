function full_SOR(A::AbstractMatrix, b::AbstractVector, maxit::Integer, omega::Float64; thres::Union{Float64, Nothing}=nothing)
    Ty = eltype(A)
    n = size(A,1)
    x = zeros(Ty, n)
    res = zeros(Ty, n)
    B = deepcopy(A)
    [B[i,i] = 0 for i = 1:n]
    D = Diagonal(diag(A))
    L = LowerTriangular(B)
    U = UpperTriangular(B)
    if typeof(thres) <: Nothing
        for i in 1:maxit
            x .= (D + omega * L) \ (omega * b - (omega * U * x + (omega - 1) * D * x))
        end

    else
        for i in 1:maxit
            x .= (D + omega * L) \ (omega * b - (omega * U * x + (omega - 1) * D * x))
            if norm(A * x - b) < thres
                break
            end

        end

    end                

    return x
end

@testset "SOR" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Test mul! functions
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        omega = 1.5
        maxit = 5
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        mul!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        mul!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        mul!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        mul!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        
        #Test ldiv! functions
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        ldiv!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega)
        @test P \ b ≈ full_SOR(A, b, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        ldiv!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        @test P \ b ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        ldiv!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega)
        @test P \ b ≈ full_SOR(A, b, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        ldiv!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        @test P \ b ≈ full_SOR(A, b, maxit, omega, thres = 10.)

        #Test one entry mul! functions
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        mul!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC,10,3,3)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        mul!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        b = rand(FC,10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega, thres = 10.)

        #Test one entry ldiv! functions
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        ldiv!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        ldiv!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega, thres = 10.)
    end
    for FC in (Float64, ComplexF64)
        #test mul! for sparse matrices
        b = rand(FC, 10)
        x = zeros(FC, 10)
        A = sprand(FC, 10, 10, .999)
        omega = 1.5
        maxit = 10
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        mul!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        mul!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        
        #Test ldiv! for sparse matrices
        b = rand(FC, 10)
        x = zeros(FC, 10)
        omega = 1.5
        maxit = 10
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        ldiv!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega)
        @test P \ b ≈ full_SOR(A, b, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        ldiv!(x, P, b)
        @test x ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        @test P \ b ≈ full_SOR(A, b, maxit, omega, thres = 10.)
        
        #test one entry mul! for sparse matrices
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        mul!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega, thres = 10.)
        
        #test one entry ldiv! for sparse matrices
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit)
        ldiv!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega)
        P = LSPreconditioners.SOR(A, omega, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ full_SOR(A, c, maxit, omega, thres = 10.)
    end
end
