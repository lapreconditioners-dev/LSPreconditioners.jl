function full_Jacobi(A::AbstractMatrix, b::AbstractVector, maxit::Integer; thres::Union{Float64, Nothing}=nothing)
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
            x = (D) \ (b - (U * x + L * x))
        end

    else
        for i in 1:maxit
            x = (D) \ (b - (U * x + L * x))
            if norm(A * x - b) < thres
                break
            end

        end

    end                

    return x
end

@testset "Jacobi" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Test mul! functions
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        maxit = 5
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        mul!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        mul!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        mul!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        mul!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit, thres = 10.)
        
        #Test ldiv! functions
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        ldiv!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit)
        @test P \ b ≈ full_Jacobi(A, b, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        ldiv!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit, thres = 10.)
        @test P \ b ≈ full_Jacobi(A, b, maxit, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        ldiv!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit)
        @test P \ b ≈ full_Jacobi(A, b, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        ldiv!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit, thres = 10.)
        @test P \ b ≈ full_Jacobi(A, b, maxit, thres = 10.)

        #Test one entry mul! functions
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        mul!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC,10,3,3)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        mul!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        b = rand(FC,10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit, thres = 10.)

        #Test one entry ldiv! functions
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        ldiv!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit, thres = 10.)
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        ldiv!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit, thres = 10.)
    end
    for FC in (Float64, ComplexF64)
        #test mul! for sparse matrices
        b = rand(FC, 10)
        x = zeros(FC, 10)
        A = sprand(FC, 10, 10, .999)
        maxit = 10
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        mul!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        mul!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit, thres = 10.)
        
        #Test ldiv! for sparse matrices
        b = rand(FC, 10)
        x = zeros(FC, 10)
        maxit = 10
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        ldiv!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit)
        @test P \ b ≈ full_Jacobi(A, b, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        ldiv!(x, P, b)
        @test x ≈ full_Jacobi(A, b, maxit, thres = 10.)
        @test P \ b ≈ full_Jacobi(A, b, maxit, thres = 10.)
        
        #test one entry mul! for sparse matrices
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        mul!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit, thres = 10.)
        
        #test one entry ldiv! for sparse matrices
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.Jacobi(A, maxit = maxit)
        ldiv!(P, b)
        @test b ≈ full_Jacobi(A, c, maxit)
        P = LSPreconditioners.Jacobi(A, maxit = maxit, thres = 10.)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
    end

end
