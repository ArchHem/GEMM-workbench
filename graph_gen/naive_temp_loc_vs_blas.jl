using Random, BenchmarkTools, LinearAlgebra, Plots, Statistics

function naive_gemm!(C::AbstractMatrix{T}, A, B) where {T}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    for j in axes(C, 2)
        for i in axes(C, 1)
            C_accum = zero(T)
            @simd for k in axes(A, 2)
            
                @inbounds @fastmath C_accum += A[i, k] * B[k, j]
            end
            C[i,j] = C_accum
        end
    end
    return nothing
end

BLAS.set_num_threads(1)
sizes = collect(1:32) .* 8

times_j_median = zeros(length(sizes))
times_j_mean = zeros(length(sizes))
times_j_std = zeros(length(sizes))

times_b_median = zeros(length(sizes))
times_b_mean = zeros(length(sizes))
times_b_std = zeros(length(sizes))

for (i, N) in enumerate(sizes)
    println(N)
    A = randn(Float32, N, N)
    B = randn(Float32, N, N)
    C = randn(Float32, N, N)
    
    bmark_j = @benchmark naive_gemm!($C, $A, $B)
    times_j_median[i] = median(bmark_j.times)
    times_j_mean[i] = mean(bmark_j.times)
    times_j_std[i] = std(bmark_j.times)

    bmark_b = @benchmark mul!($C, $A, $B)
    times_b_median[i] = median(bmark_b.times)
    times_b_mean[i] = mean(bmark_b.times)
    times_b_std[i] = std(bmark_b.times)
end

p1 = plot(xlabel="Matrix Size (N)",
         ylabel="Time (ms)",
         title="Naive Benchmark vs BLAS mul!",
         legend=:topleft, yscale = :log)

scatter!(p1, sizes, times_j_mean ./ 1e6,
         yerror=times_j_std ./ 1e6,
         label="Naive (Mean ± SD)",
         color=:purple, markersize = 2.2)

scatter!(p1, sizes, times_b_mean ./ 1e6,
         yerror=times_b_std ./ 1e6,
         label="BLAS mul! (Mean ± SD)",
         color=:green, markersize = 2.2)

ratio = times_j_mean ./ times_b_mean
p2 = plot(xlabel="Matrix Size (N)",
          ylabel="Ratio (Naive / BLAS)",
          legend=false)

scatter!(p2, sizes, ratio, color=:blue, markersize=2.5)
hline!(p2, [1.0], linestyle=:dash, color=:black)


final_plot = plot(p1, p2, layout=(2, 1), link=:x, dpi = 600)

savefig(final_plot, "./media/naive_temp_loc_vs_blas_comp.png")