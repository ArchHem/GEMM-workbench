include("../src/GEMM.jl")
using .GEMM
using Plots

N_y, N_x = 32, 16
J = 4
K_macro = div(N_y * N_x, J)
padding = 16

m = zeros(Int64, N_y, N_x)
m .= LinearIndices(m)
buffer = zeros(Int64, J, K_macro)

GEMM.pack_B!(buffer, m, GEMM.TileControl(8, 1, 1, 8, 4, 2))

canvas_height = N_y + padding + J
canvas_width = K_macro
final_canvas = fill(NaN, canvas_height, canvas_width)

m_col_start = (canvas_width - N_x) ÷ 2 + 1
m_col_end = m_col_start + N_x - 1
final_canvas[1:N_y, m_col_start:m_col_end] = m

buffer_row_start = N_y + padding + 1
buffer_row_end = buffer_row_start + J - 1
final_canvas[buffer_row_start:buffer_row_end, 1:K_macro] = buffer

p = heatmap(final_canvas,
            c=:viridis,
            aspect_ratio=:equal,
            framestyle=:none,
            colorbar=false,
            background_color=:black,
            size=(1000, 600),
            yflip=true, dpi = 600)

function add_borders_and_ticks!(plt, rows, cols; color = :red)
    row_start, row_end = rows
    col_start, col_end = cols
    
    border_shape = [(col_start - 0.5, row_start - 0.5), (col_end + 0.5, row_start - 0.5), (col_end + 0.5, row_end + 0.5), (col_start - 0.5, row_end + 0.5)]
    plot!(plt, border_shape, seriestype=:shape, fillalpha=0, linecolor=color, linewidth=1.5, label="")

    for i in col_start:col_end-1; plot!(plt, [i + 0.5, i + 0.5], [row_start - 0.5, row_end + 0.5], color=color, linewidth=0.2, label=""); end
    for i in row_start:row_end-1; plot!(plt, [col_start - 0.5, col_end + 0.5], [i + 0.5, i + 0.5], color=color, linewidth=0.2, label=""); end
end

function add_sub_matrix_highlight!(plt, rows, cols, label_text; color=:cyan, text_size=12)
    row_start, row_end = rows
    col_start, col_end = cols
    border_shape = [(col_start - 0.5, row_start - 0.5), (col_end + 0.5, row_start - 0.5), (col_end + 0.5, row_end + 0.5), (col_start - 0.5, row_end + 0.5)]
    plot!(plt, border_shape, seriestype=:shape, fillalpha=0.1, fillcolor=color, linecolor=color, linewidth=2.5, label="")
    if !isempty(label_text)
        annotate!(col_end, row_start, text(label_text, color, :right, :top, text_size))
    end
end

add_borders_and_ticks!(p, (1, N_y), (m_col_start, m_col_end), color=:cyan)
add_borders_and_ticks!(p, (buffer_row_start, buffer_row_end), (1, K_macro), color=:cyan)

annotate!(p, m_col_end, 1, text("L2", :cyan, :right, :top, 12))
annotate!(p, K_macro, buffer_row_start, text("L2", :cyan, :right, :top, 12))

src_rows = (1, N_y)
src_cols = (m_col_start, m_col_start + J - 1)
dest_rows = (buffer_row_start, buffer_row_end)
dest_cols = (1, N_y)

add_sub_matrix_highlight!(p, src_rows, src_cols, "L1"; color=:orange, text_size=9)
add_sub_matrix_highlight!(p, dest_rows, dest_cols, "L1"; color=:orange, text_size=9)

src_tl = (src_cols[1] - 0.5, src_rows[1] - 0.5)
src_tr = (src_cols[2] + 0.5, src_rows[1] - 0.5)
src_bl = (src_cols[1] - 0.5, src_rows[2] + 0.5)
src_br = (src_cols[2] + 0.5, src_rows[2] + 0.5)

dest_tl = (dest_cols[1] - 0.5, dest_rows[1] - 0.5)
dest_tr = (dest_cols[2] + 0.5, dest_rows[1] - 0.5)
dest_bl = (dest_cols[1] - 0.5, dest_rows[2] + 0.5)
dest_br = (dest_cols[2] + 0.5, dest_rows[2] + 0.5)

for (p1_corner, p2_corner) in [(src_tl, dest_tl), (src_tr, dest_bl), (src_bl, dest_tr), (src_br, dest_br)]
    plot!(p, [p1_corner[1], p2_corner[1]], [p1_corner[2], p2_corner[2]], color=:white, style=:dash, label="")
end

arrow_y_start = N_y + 1
arrow_y_end = N_y + padding
arrow_x = canvas_width / 2
annotate!(p, arrow_x, arrow_y_end - 2, text(" packing ↓", :white, :center, 12))
savefig(p, "./media/packing_visul.png")