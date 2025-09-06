module GEMM

using Base.Iterators
using SIMD

struct TileControl{K_unroll, K_accum_spill, K_load_spill, I, J, register_width}
    function TileControl(K_unroll::Int, K_accum_spill::Int, K_load_spill::Int, I::Int, J::Int, register_width::Int)

        if rem(K_unroll, K_accum_spill) != 0
            throw(ArgumentError("The unroll length must be divisible by the accumulator spill length."))
        end

        if rem(K_unroll, K_load_spill) != 0
            throw(ArgumentError("The unroll length must be divisible by the load spill length."))
        end

        if rem(I, register_width) != 0
            throw(ArgumentError("The tile dimensions must be divisible by the register width."))
        end

        if rem(J, register_width) != 0
            throw(ArgumentError("The tile dimensions must be divisible by the register width."))
        end



        return new{K_unroll, K_accum_spill, K_load_spill, I, J, register_width}()
    end
end

#TODO: Complete the bellow, using SIMD, for 16 x 4
#TODO: fix 2-based indexing in inner loops in GEMM

@generated function microkernel_SIMD!(C::AbstractMatrix{T}, ATile, BTile, K_macro,
    tile::TileControl{K_unroll, K_accum_spill, K_load_spill, I, J, register_width}) where 
    {T, K_unroll, K_accum_spill, K_load_spill, I, J, register_width}

    #define loader lane

    lane = VecRange{register_width}(1)

    #lets define variables
    I_N_vars = div(I, register_width)
    J_N_vars = div(J, register_width)

    c_vecs = [gensym("c_acc_$(i)_$(j)_$(k)") for i in 1:I_N_vars, j in 1:J, k in 1:K_accum_spill]
    a_vecs = [gensym("a_var_$(i)_$(k)") for i in 1:I_N_vars, k in 1:K_load_spill]
    b_vecs = [gensym("b_var_$(j)_$(k)") for j in 1:J_N_vars, k in 1:K_load_spill]

    #initalize the accummulators

    init_expr = [:( $(c_vecs[i,j,k]) = zero(Vec{register_width,T})) for i in 1:I_N_vars, j in 1:J, k in 1:K_accum_spill]

    #unroll along K
    
    k_loop_var = gensym("k_loop")
    

    standard_accum = []

    for k_unrolled in 1:K_unroll
        #identifies the current spilling "step"
        accum_spill_idx = rem(k_unrolled - 1, K_accum_spill) + 1
        load_spill_idx = rem(k_unrolled - 1, K_load_spill) + 1

        a_loads_local = [:( @inbounds $(a_vecs[i+1, load_spill_idx]) =
            ATile[$(i*register_width + lane), $k_loop_var + $k_unrolled]) for i in 0:I_N_vars-1]
        
        b_loads_local = [:( @inbounds $(b_vecs[j+1, load_spill_idx]) =
            BTile[$(j*register_width + lane), $k_loop_var + $k_unrolled]) for j in 0:J_N_vars-1]
        
        c_accum_local = [:( @inbounds @fastmath $(c_vecs[i, j+1, accum_spill_idx]) += 
            $(a_vecs[i, load_spill_idx]) * getindex($(b_vecs[div(j, register_width)+1, load_spill_idx]), $(mod(j, register_width) + 1)) ) for i in 1:I_N_vars, j in 0:J-1]
        push!(standard_accum, a_loads_local...)
        push!(standard_accum, b_loads_local...)
        push!(standard_accum, c_accum_local...)
    end

    reduce_instr = []

    for i in 1:I_N_vars
        for j in 1:J
            for a in 2:K_accum_spill
                #accumulate all rotatory register into the first one, ghopefully this translates to assembly...
                push!(reduce_instr, :(@inbounds @fastmath $(c_vecs[i, j, 1]) += $(c_vecs[i, j, a])))
            end
        end
    end

    store_instr = []

    for i in 0:I_N_vars-1
        for j in 1:J
            
            push!(store_instr, :(@inbounds @fastmath C[$lane + $(i*register_width), $j] += $(c_vecs[i+1, j, 1])))
            
        end
    end

    res = quote 
        $(init_expr...)
        for $k_loop_var in 0:K_unroll:(K_macro-1)
            $(standard_accum...)
        end
        
        $(reduce_instr...)
        $(store_instr...)
        nothing
    end
    return res


end


function pack_B!(buffer, b_slice, tile::TileControl{K_unroll, K_accum_spill, K_load_spill, I, J, register_width}) where {K_unroll, K_accum_spill, K_load_spill, I, J, register_width}

    #reorder B into the buffer st that the microkernel access becomes continious.
    #buffer has size (J, K_macro), input is arbitary shaped:
    #The number of elements mathc, I.e. (J*K_macro) = (K_input*J_input)
    K_slice = size(b_slice, 1)
    j_counter = 0
    @inbounds for j_tile in partition(axes(b_slice, 2), J)
        for k_tile in partition(1:K_slice, K_unroll)
            for j in j_tile
                j_index = mod1(j, J)
                @simd for k in k_tile
                    k_index = k + (K_slice * j_counter)
                    @inbounds buffer[j_index, k_index] = b_slice[k, j]
                end
            end
        end
        j_counter += 1
    end

    return nothing
end

function pack_A!(buffer, a_slice, tile::TileControl{K_unroll, K_accum_spill, K_load_spill, I, J, register_width}) where {K_unroll, K_accum_spill, K_load_spill, I, J, register_width}
    K_slice = size(a_slice, 2)
    
    for k in 1:K_slice
        @simd for i in axes(a_slice, 1)
            i_index = mod1(i, I)
            k_index = k + K_slice * div(i-1, I)
            @inbounds buffer[i_index,  k_index] = a_slice[i, k]
        end
    end
    
    return nothing
end

#Cache limits depend on constituent elements.
function fadd_GEMM!(C::Matrix{Float32}, A, B; 
        tile::TileControl{K_unroll, K_accum_spill, K_load_spill, I, J, register_width} = TileControl(8, 1, 1, 16, 4, 4), 
        i_partition = 2304, k_partition = 1600, j_partition = 984) where {K_unroll, K_accum_spill, K_load_spill, I, J, register_width}
    
    #decrease K_partition in case L1 gets contested...
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    I_total = size(C, 1)
    J_total = size(C, 2)
    K_total = size(A, 2)

    @assert rem(I_total, I) == 0
    @assert rem(J_total, J) == 0
    @assert rem(K_total, K_unroll) == 0

    @assert rem(i_partition, I) == 0
    @assert rem(j_partition, J) == 0
    @assert rem(k_partition, K_unroll) == 0

    #size of B_buffer should be equal to that of the B slice.
    B_buffer = Matrix{Float32}(undef, J, div(min(k_partition, K_total)*min(j_partition, J_total), J))
    A_buffer = Matrix{Float32}(undef, I, div(min(k_partition, K_total)*min(i_partition, I_total), I))
    

    @inbounds for i_macro in partition(1:I_total, i_partition)
        for k_macro in partition(1:K_total, k_partition)
            A_slice = @inbounds @views A[i_macro, k_macro]
            pack_A!(A_buffer, A_slice, tile)
            for j_macro in partition(1:J_total, j_partition)
                #we repack B every iteration of i - this is unnesseary, but unavoidable.
                @inbounds B_slice = @views B[k_macro, j_macro]
                pack_B!(B_buffer, B_slice, tile)

                i_counter = 0
                for i_micro in partition(i_macro, I)
                    @inbounds A_view = @views A_buffer[1:I, (length(k_macro) .* (i_counter):(i_counter + 1)) .+ 1]
                    j_counter = 0
                    for j_micro in partition(j_macro, J)
                        @inbounds C_view = @views C[i_micro, j_micro]
                        if first(k_macro) == 1
                            C_view .= 0f0
                        end
                        @inbounds B_view = @views B_buffer[1:J, (length(k_macro) .* (j_counter):(j_counter + 1)) .+ 1 ]
                        microkernel_SIMD!(C_view, A_view, B_view, length(k_macro), tile)
                        j_counter += 1
                    end
                    i_counter += 1
                end

            end
        end
    end

    return nothing

end

export microkernel_SIMD!, TileControl, fadd_GEMM!
end