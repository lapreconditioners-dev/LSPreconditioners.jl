@testset "SPAI" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Testing that approximate inverse when SPAI matches solution to problem when sparsity pattern is full mat
        #Test the mul! function
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Start with DenseMatrix
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
        
        #Test the ldiv! function
        b = rand(FC, 10)
        x = zeros(FC, 10)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(x, P, b)
        @test x ≈ A \ b 
        @test P \ b ≈ A \ b 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(x, P, b)
        @test x ≈ A \ b 
        @test P \ b ≈ A \ b 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(x, P, b)
        @test x ≈ A \ b 
        @test P \ b ≈ A \ b 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(x, P, b)
        @test x ≈ A \ b 
        @test P \ b ≈ A \ b 
        
        #Test one entry mul! functions
        b = rand(FC, 10)
        c = deepcopy(b)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(P, b)
        @test b ≈ A \ c 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ A \ c 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        Z = sprand(FC, 10, 10, 1.)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(P, b)
        @test b ≈ A \ c 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        b = rand(FC, 10)
        c = deepcopy(b)
        mul!(P, b)
        @test b ≈ A \ c 
        
        #Test one entry ldiv! functions
        b = rand(FC, 10)
        c = deepcopy(b)
        #Start with DenseMatrix
        A = rand(FC, 10, 10)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(P, b)
        @test b ≈ A \ c 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ A \ c 
        #Next to BandedMatrix
        A = brand(FC, 10, 3, 3)
        Z = sprand(FC, 10, 10, 1.)
        b = rand(FC, 10)
        c = deepcopy(b)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(P, b)
        @test b ≈ A \ c 
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        b = rand(FC, 10)
        c = deepcopy(b)
        ldiv!(P, b)
        @test b ≈ A \ c 
    end
    for FC in (Float64, ComplexF64)
        #Finish with sparsematrices
        #Test mul!
        b = rand(FC, 10)
        x = zeros(FC, 10)
        A = sprand(FC, 10, 10, .99)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(x, P, b)
        @test x ≈ A \ b
       
        #Test ldiv!
        b = rand(FC, 10)
        x = zeros(FC, 10)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(x, P, b)
        @test x ≈ A \ b
        @test P \ b ≈ A \ b
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(x, P, b)
        @test x ≈ A \ b
        @test P \ b ≈ A \ b
        
        #Test single entry mul!
        b = rand(FC, 10)
        c = deepcopy(b)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(P, b)
        @test b ≈ A \ c
        b = rand(FC, 10)
        c = deepcopy(b)
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        mul!(P, b)
        @test b ≈ A \ c
        
        #Test single entry ldiv!
        b = rand(FC, 10)
        c = deepcopy(b)
        Z = sprand(FC, 10, 10, 1.)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(P, b)
        @test b ≈ A \ c
        b = rand(FC, 10)
        c = deepcopy(b)
        Z = brand(FC, 10, 9, 9)
        P = LSPreconditioners.SPAI(A, Z)
        ldiv!(P, b)
        @test b ≈ A \ c
    end
end
