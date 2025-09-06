include("../src/GEMM.jl")
using .GEMM
using Test, Random

@testset "Microkernel" begin

    Random.seed!(1234)

    I, J = 12, 8

    for K_macro in (16, 32, 64, 128, 512)
        for K_unroll in (4, 8, 16)
            
            for k_accum_spill in (1, 2)
                for k_load_spill in (1, 2)
                    tile = TileControl(K_unroll, k_accum_spill, k_load_spill, I, J, 4)
                    a = randn(Float32, I, K_macro)
                    b = randn(Float32, J, K_macro)
                    c = zeros(Float32, I, J)
                    microkernel_SIMD!(c, a, b, K_macro, tile)
                    c_target = a * b'
                    @test isapprox(c_target, c)
                end
            end
            
        end
    end

end

@testset "GEMM" begin
    Random.seed!(1234)

    for I in (256, 800, 1024, 4096)
        for J in (256, 400, 2048, 4096)
            for K in (8, 256, 4096)
                
                
                C = randn(Float32, I, J)
                A = randn(Float32, I, K)
                B = randn(Float32, K, J)
                fadd_GEMM!(C, A, B)

                @test isapprox(C, A*B)
            end
        end
    end

    
end