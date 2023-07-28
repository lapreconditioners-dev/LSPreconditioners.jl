
function SolveBlocks(A::AbstractMatrix, b::AbstractVector, bsize::Integer)
    n = length(b)
    x = zeros(n)
    remB = rem(m, blocksize)
    nblocks = div(m, blocksize) + (remB == 0 ? 0 : 1)
    bsizes = Array{Int}(undef, nblocks)
    T = eltype(A)
    for i in 1:nblocks-1
        bsizes[i] = blocksize
    end

    bsizes[nblocks] = remB == 0 ? blocksize : remB
    endp = 0
    for i in 1:nblocks
        startp = endp + 1
        endp = startp + blocksize[i]
        x[startp:endp] = A[startp:endp, startp:endp] \ b[startp:endp]
    end
    return x
end

@testset "blockjacobi" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC,10)
        x = zeros(FC,10)
        #Satrt with DenseMatrix
        A = rand(FC,10,10)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x = SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x = SolveBlocks(A,b,3)
        #Next to BandedMatrix
        A = rand(FC,10,3,3)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x = SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x = SolveBlocks(A,b,3)
        #Finish with sparsematrices
        A = sprand(FC,10,10,.9)
        P = LSPreconditioners.BlockJacobi(A,2)
        mul!(x,P,b)
        @test x = SolveBlocks(A,b,2)
        P = LSPreconditioners.BlockJacobi(A,3)
        mul!(x,P,b)
        @test x = SolveBlocks(A,b,3)
    end
end
