@testset "SPAI" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b 
    end
    for FC in (Float64, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x,P,b)
        @test x ≈ A \ b
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b
    end
end
