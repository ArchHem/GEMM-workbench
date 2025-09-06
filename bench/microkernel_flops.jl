include("../src/GEMM.jl")
using .GEMM
using BenchmarkTools, LinearAlgebra, Random, Plots

I, J = 16, 4
K_macro = 1600

Random.seed!(1)
A = randn(Float32, I, K_macro)
B = randn(Float32, J, K_macro)
C = zeros(Float32, I, J)

tile = TileControl(8, 1, 1, I, J, 4)
mkernel_bmark = @benchmark microkernel_SIMD!($C, $A, $B, $K_macro, $tile)

function get_gflops(bmark, i, j, k)
    gflops = (2 * i * j * k) ./ bmark.times
    return gflops
end

mkernel_gflops = get_gflops(mkernel_bmark, I, J, K_macro)

p = histogram(mkernel_gflops, bins = 400,
          label="Microkernel GFLOPs",
          color=:orange,
          xlabel="Performance (GFLOPS)",
          ylabel="Counts",
          title="GFLOPs Distribution of Microkernel calls",
          legend=:topleft, dpi = 600, xlim = (90, 110))

vline!(p, [median(mkernel_gflops)], color=:green, label = "Median GFLOPs: $(round(median(mkernel_gflops); digits = 3))")
vline!(p, [mean(mkernel_gflops)], color=:blue, label = "Mean GFLOPs: $(round(mean(mkernel_gflops); digits = 3))")
savefig(p, "./media/microkernel_flops.png")