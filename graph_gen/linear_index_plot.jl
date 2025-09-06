using Plots

Nx = 32
Ny = 16
padding = 8

total_width = Nx + padding + Ny + padding + Ny
total_height = Nx

canvas = fill(NaN, total_height, total_width)

y_offset_A = (total_height - Ny) ÷ 2
row_start_A = y_offset_A + 1
row_end_A = row_start_A + Ny - 1
col_start_A = 1
col_end_A = Nx
canvas[row_start_A:row_end_A, col_start_A:col_end_A] = LinearIndices((Ny, Nx))

col_start_B = col_end_A + padding + 1
col_end_B = col_start_B + Ny - 1
canvas[1:total_height, col_start_B:col_end_B] = LinearIndices((Nx, Ny))

col_start_C = col_end_B + padding + 1
col_end_C = col_start_C + Ny - 1
canvas[row_start_A:row_end_A, col_start_C:col_end_C] = LinearIndices((Ny, Ny))

p = heatmap(canvas,
            c=:viridis,
            aspect_ratio=:equal,
            framestyle=:none,
            colorbar=false,
            background_color=:black, dpi = 600)

function add_borders_and_ticks!(plt, rows, cols)
    row_start, row_end = rows
    col_start, col_end = cols

    border_shape = [
        (col_start - 0.5, row_start - 0.5),
        (col_end + 0.5, row_start - 0.5),
        (col_end + 0.5, row_end + 0.5),
        (col_start - 0.5, row_end + 0.5)
    ]
    plot!(plt, border_shape, seriestype=:shape, fillalpha=0, linecolor=:red, linewidth=1.5, label="")

    for i in col_start:col_end-1
        plot!(plt, [i + 0.5, i + 0.5], [row_start - 0.5, row_end + 0.5], color=:red, linewidth=0.2, label="")
    end
    for i in row_start:row_end-1
        plot!(plt, [col_start - 0.5, col_end + 0.5], [i + 0.5, i + 0.5], color=:red, linewidth=0.2, label="")
    end
end

add_borders_and_ticks!(p, (row_start_A, row_end_A), (col_start_A, col_end_A))
add_borders_and_ticks!(p, (1, total_height), (col_start_B, col_end_B))
add_borders_and_ticks!(p, (row_start_A, row_end_A), (col_start_C, col_end_C))

mul_pos_x = col_end_A + padding / 2
eq_pos_x = col_end_B + padding / 2
symbol_pos_y = total_height / 2

annotate!(mul_pos_x, symbol_pos_y, text("*", :white, 24))
annotate!(eq_pos_x, symbol_pos_y, text("=", :white, 24))

savefig(p, "./media/linear_index_matmul.png")