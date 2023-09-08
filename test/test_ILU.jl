@testset "ILU" begin
    for FC in (Float32, Float64, ComplexF32, ComplexF64)
        #Test for mul!
        b = rand(FC, 20)
        xw = zeros(FC, 20)
        xt = zeros(FC, 20)
        #Start with ILUZero
        A = sprand(FC, 20, 20, .9999)
        Pw = LSPreconditioners.ILU(A)
        Pt = ilu0(A)
        mul!(xw, Pw, b)
        ldiv!(xt, Pt, b)
        @test xw ≈ xt
        #Start with threshold ILU
        Pw = LSPreconditioners.ILU(A, 3)
        Pt = ilu(A, τ=3)
        mul!(xw, Pw, b)
        ldiv!(xt, Pt, b)
        @test xw ≈ xt

        #Test for ldiv!
        b = rand(FC, 20)
        xw = zeros(FC, 20)
        xt = zeros(FC, 20)
        #Start with ILUZero
        A = sprand(FC, 20, 20, .9999)
        Pw = LSPreconditioners.ILU(A)
        Pt = ilu0(A)
        ldiv!(xw, Pw, b)
        ldiv!(xt, Pt, b)
        @test xw ≈ xt
        #Start with threshold ILU
        Pw = LSPreconditioners.ILU(A, 3)
        Pt = ilu(A, τ=3)
        ldiv!(xw, Pw, b)
        ldiv!(xt, Pt, b)
        @test xw ≈ xt
        
        #Test for one entry mul!
        b = rand(FC, 20)
        c = deepcopy(b)
        #Start with ILUZero
        A = sprand(FC, 20, 20, .9999)
        Pw = LSPreconditioners.ILU(A)
        Pt = ilu0(A)
        mul!(Pw, b)
        ldiv!(Pt, c)
        @test b ≈ c
        #Start with threshold ILU
        b = rand(FC, 20)
        c = deepcopy(b)
        Pw = LSPreconditioners.ILU(A, 3)
        Pt = ilu(A, τ=3)
        ldiv!(Pw, b)
        ldiv!(Pt, c)
        @test b ≈ c

        #Test for one entry ldiv!
        b = rand(FC, 20)
        c = deepcopy(b)
        #Start with ILUZero
        A = sprand(FC, 20, 20, .9999)
        Pw = LSPreconditioners.ILU(A)
        Pt = ilu0(A)
        mul!(Pw, b)
        ldiv!(Pt, c)
        @test b ≈ c
        #Start with threshold ILU
        b = rand(FC, 20)
        c = deepcopy(b)
        Pw = LSPreconditioners.ILU(A, 3)
        Pt = ilu(A, τ=3)
        ldiv!(Pw, b)
        ldiv!(Pt, c)
        @test b ≈ c
    end
end
