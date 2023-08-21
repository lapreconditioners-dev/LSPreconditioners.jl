#Function to apply the inverse of the diagonal of a matrix to a vector
function DiagApply(A, b)
    c = deepcopy(b)
    ln = length(c)
    dg = diag(A)
    for i = 1:ln
        c[i] = b[i] / (dg[i] != 0 ? dg[i] : 1)
    end
    return c
end

@testset "Diagonal" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(x, P, b)
        @test x ≈ DiagApply(A, b) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(x, P, b)
        @test x ≈ DiagApply(A, b) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(x, P, b)
        @test x ≈ DiagApply(A, b) 

        #Same tests for ldiv
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        ldiv!(x, P, b)
        @test x ≈ DiagApply(A, b) 
        @test P \ b ≈ DiagApply(A, b) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        ldiv!(x, P, b)
        @test x ≈ DiagApply(A, b) 
        @test P \ b ≈ DiagApply(A, b) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        ldiv!(x, P, b)
        @test x ≈ DiagApply(A, b) 
        @test P \ b ≈ DiagApply(A, b) 

        #Same test for one entry mul!
        b = rand(FC, 10)
        c = deepcopy(b)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        mul!(P, b)
        @test b ≈ DiagApply(A, c) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ DiagApply(A, c) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ DiagApply(A, c) 
        
        #Same test for one entry ldiv!
        b = rand(FC, 10)
        c = deepcopy(b)
        x = zeros(FC, 10)
        #Satrt with DenseMatrix
        A = rand(FC, 10, 10)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        ldiv!(P, b)
        @test b ≈ DiagApply(A, c) 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ DiagApply(A, c) 
        #Finish with sparsematrices
        A = sprand(FC, 10, 10, .99)
        P = LSPreconditioners.DiagonalPreconditioner(A)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ DiagApply(A, c) 
    end
end

