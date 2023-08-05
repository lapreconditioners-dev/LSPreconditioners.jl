@testset "ILU" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        b = rand(FC, 20)
        xw = zeros(FC, 20)
        xt = zeros(FC, 20)
        #Start with ILUZero
        A = sprand(FC, 20, 20, .8)
        Pw = LSPreconditioners.ILU(A)
        Pt = ilu0(A)
        mul!(xw, Pw, b)
        ldiv!(xt, Pt, b)
        @test xw ≈ xt
        #Start with threshold ILU
        Pw = LSPreconditioners.ILU(A, t=3)
        Pt = ilu(A, τ=3)
        mul!(xw, Pw, b)
        ldiv!(xt, Pt, b)
        @test xw ≈ xt
    end
end
