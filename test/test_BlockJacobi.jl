
function SolveBlocks(A::AbstractMatrix, b::AbstractVector, blocksize::Integer)
    T = eltype(A)
    m = length(b)
    x = zeros(T,m)
    remB = rem(m, blocksize)
    nblocks = div(m, blocksize) + (remB == 0 ? 0 : 1)
    bsizes = Array{Int}(undef, nblocks)
    for i in 1:nblocks-1
        bsizes[i] = blocksize
    end

    bsizes[nblocks] = remB == 0 ? blocksize : remB
    endp = 0
    for i in 1:nblocks
        startp = endp + 1
        endp = endp + bsizes[i]
        x[startp:endp] = A[startp:endp, startp:endp] \ b[startp:endp]
    end
    return x
end

@testset "blockjacobi" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Test mul! functions
        b = rand(FC,10)
        x = zeros(FC,10)
        #Start with DenseMatrix
        A = rand(FC,10,10)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,3)
        #Next to BandedMatrix
        A = brand(FC,10,3,3)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,3)
        
        #Test ldiv! functions
        b = rand(FC,10)
        x = zeros(FC,10)
        #Start with DenseMatrix
        A = rand(FC,10,10)
        P = LSPreconditioners.BlockJacobi(A,2)
        ldiv!(x,P,b)
        @test x ≈ SolveBlocks(A,b,2)
        @test P \ b ≈ SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        ldiv!(x,P,b)
        @test x ≈ SolveBlocks(A,b,3)
        @test P \ b ≈ SolveBlocks(A,b,3)
        #Next to BandedMatrix
        A = brand(FC,10,3,3)
        P = LSPreconditioners.BlockJacobi(A,2)
        ldiv!(x,P,b)
        @test x ≈ SolveBlocks(A,b,2)
        @test P \ b ≈ SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        ldiv!(x,P,b)
        @test x ≈ SolveBlocks(A,b,3)
        @test P \ b ≈ SolveBlocks(A,b,3)

        #Test one entry mul! functions
        b = rand(FC,10)
        c = deepcopy(b)
        #Start with DenseMatrix
        A = rand(FC,10,10)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(P,b)
        @test b ≈ SolveBlocks(A,c,2)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(P,b)
        @test b ≈ SolveBlocks(A,c,3)
        #Next to BandedMatrix
        A = brand(FC,10,3,3)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(P,b)
        @test b ≈ SolveBlocks(A,c,2)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(P,b)
        @test b ≈ SolveBlocks(A,c,3)

        #Test one entry ldiv! functions
        b = rand(FC,10)
        c = deepcopy(b)
        #Start with DenseMatrix
        A = rand(FC,10,10)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,2)
        ldiv!(P,b)
        @test b ≈ SolveBlocks(A,c,2)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,3)
        ldiv!(P,b)
        @test b ≈ SolveBlocks(A,c,3)
        #Next to BandedMatrix
        A = brand(FC,10,3,3)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,2)
        ldiv!(P,b)
        @test b ≈ SolveBlocks(A,c,2)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,3)
        ldiv!(P,b)
        @test b ≈ SolveBlocks(A,c,3)
    end
    for FC in (Float64, ComplexF64)
        #test mul! for sparse matrices
        b = rand(FC,10)
        x = zeros(FC,10)
        #Finish with sparsematrices
        A = sprand(FC,10,10,.999)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,3)
        
        #Test ldiv!
        b = rand(FC,10)
        x = zeros(FC,10)
        #Finish with sparsematrices
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,2)
        @test P \ b ≈ SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x ≈ SolveBlocks(A,b,3)
        @test P \ b ≈ SolveBlocks(A,b,3) 
        
        #test one entry mul! for sparse matrices
        b = rand(FC,10)
        c = deepcopy(b)
        #Finish with sparsematrices
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(P,b)
        @test b ≈ SolveBlocks(A,c,2)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(P,b)
        @test b ≈ SolveBlocks(A,c,3)
        
        #test one entry ldiv! for sparse matrices
        b = rand(FC,10)
        c = deepcopy(b)
        #Finish with sparsematrices
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,2)
        ldiv!(P,b)
        @test b ≈ SolveBlocks(A,c,2)
        b = rand(FC,10)
        c = deepcopy(b)
        P = LSPreconditioners.BlockJacobi(A,3)
        ldiv!(P,b)
        @test b ≈ SolveBlocks(A,c,3)
    end
end
