@testset "Diagonal" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(x, P, b)
        @test x ≈ b ./ diag(A) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(x, P, b)
        @test x ≈ b ./ diag(A) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(x,P,b)
        @test x ≈ b ./ diag(A) 
    end
end

