using Plots

Nx = 32
Ny = 32
padding = 8
zoom_gap = 20
zoom_scale = 2

k_tile_size = 16
l3_height, l3_width = 12, k_tile_size
l2_height, l2_width = k_tile_size, 8
reg_height, reg_width = 4, 2

total_width_p1 = Nx + padding + Ny + padding + Ny
total_height_p1 = Nx

scaled_l3_height = l3_height * zoom_scale
scaled_l3_width = l3_width * zoom_scale
scaled_l2_height = l2_height * zoom_scale
scaled_l2_width = l2_width * zoom_scale
scaled_padding = padding * zoom_scale
scaled_reg_height = reg_height * zoom_scale
scaled_reg_width = reg_width * zoom_scale
zoom_total_width_p2 = scaled_l3_width + scaled_padding + scaled_l2_width + scaled_padding + scaled_l2_width
zoom_total_height_p2 = scaled_l2_height

final_canvas_height = total_height_p1 + zoom_gap + zoom_total_height_p2
final_canvas_width = max(total_width_p1, zoom_total_width_p2)
final_canvas = fill(NaN, final_canvas_height, final_canvas_width)

x_offset_p2 = (final_canvas_width - zoom_total_width_p2) ÷ 2
y_offset_p2 = total_height_p1 + zoom_gap

y_offset_A = (total_height_p1 - Ny) ÷ 2; row_start_A = y_offset_A + 1; row_end_A = row_start_A + Ny - 1
col_start_A = 1; col_end_A = Nx
final_canvas[row_start_A:row_end_A, col_start_A:col_end_A] = LinearIndices((Ny, Nx))
col_start_B = col_end_A + padding + 1; col_end_B = col_start_B + Ny - 1
final_canvas[1:total_height_p1, col_start_B:col_end_B] = LinearIndices((Nx, Ny))
col_start_C = col_end_B + padding + 1; col_end_C = col_start_C + Ny - 1
final_canvas[row_start_A:row_end_A, col_start_C:col_end_C] = LinearIndices((Ny, Ny))

linear_indices_A_full = LinearIndices((Ny, Nx)); tile_A = linear_indices_A_full[1:l3_height, 1:l3_width]
linear_indices_B_full = LinearIndices((Nx, Ny)); tile_B = linear_indices_B_full[1:l2_height, 1:l2_width]
linear_indices_C_full = LinearIndices((Ny, Ny)); tile_C = linear_indices_C_full[1:l3_height, 1:l2_width]

final_canvas[y_offset_p2+1:y_offset_p2+scaled_l3_height, x_offset_p2+1:x_offset_p2+scaled_l3_width] = repeat(tile_A, inner=(zoom_scale, zoom_scale))
z_col_start_B_p2 = scaled_l3_width + scaled_padding + 1
z_col_end_B_p2 = z_col_start_B_p2 + scaled_l2_width - 1
final_canvas[y_offset_p2+1:y_offset_p2+scaled_l2_height, x_offset_p2+z_col_start_B_p2:x_offset_p2+z_col_end_B_p2] = repeat(tile_B, inner=(zoom_scale, zoom_scale))
z_col_start_C_p2 = z_col_end_B_p2 + scaled_padding + 1
z_col_end_C_p2 = z_col_start_C_p2 + scaled_l2_width - 1
final_canvas[y_offset_p2+1:y_offset_p2+scaled_l3_height, x_offset_p2+z_col_start_C_p2:x_offset_p2+z_col_end_C_p2] = repeat(tile_C, inner=(zoom_scale, zoom_scale))

p = heatmap(final_canvas, c=:viridis, aspect_ratio=:equal, framestyle=:none, colorbar=false, background_color=:black, size=(1200, 1000), yflip=true, dpi = 600)

function add_borders_and_ticks!(plt, rows, cols; color = :red, offset = (0,0), scale=1)
    x_off, y_off = offset
    row_start, row_end = rows[1]+y_off, rows[2]+y_off
    col_start, col_end = cols[1]+x_off, cols[2]+x_off
    
    border_shape = [(col_start - 0.5, row_start - 0.5), (col_end + 0.5, row_start - 0.5), (col_end + 0.5, row_end + 0.5), (col_start - 0.5, row_end + 0.5)]
    plot!(plt, border_shape, seriestype=:shape, fillalpha=0, linecolor=color, linewidth=1.5, label="")

    num_orig_cols = (cols[2] - cols[1] + 1) ÷ scale
    for i in 1:num_orig_cols - 1
        line_x = col_start + i * scale - 0.5
        plot!(plt, [line_x, line_x], [row_start - 0.5, row_end + 0.5], color=color, linewidth=0.2, label="")
    end
    num_orig_rows = (rows[2] - rows[1] + 1) ÷ scale
    for i in 1:num_orig_rows - 1
        line_y = row_start + i * scale - 0.5
        plot!(plt, [col_start - 0.5, col_end + 0.5], [line_y, line_y], color=color, linewidth=0.2, label="")
    end
end

function add_sub_matrix_highlight!(plt, base_rows, base_cols, sub_dims, label_text; color=:cyan, text_size=12, offset=(0,0))
    x_off, y_off = offset; sub_height, sub_width = sub_dims; row_start = base_rows[1]+y_off; col_start = base_cols[1]+x_off
    row_end = row_start + sub_height - 1; col_end = col_start + sub_width - 1
    border_shape = [(col_start - 0.5, row_start - 0.5), (col_end + 0.5, row_start - 0.5), (col_end + 0.5, row_end + 0.5), (col_start - 0.5, row_end + 0.5)]
    plot!(plt, border_shape, seriestype=:shape, fillalpha=0, linecolor=color, linewidth=2.5, label="")
    if !isempty(label_text); if label_text == "L3"; annotate!(col_end, row_end, text(label_text, color, :right, :bottom, text_size)); else; annotate!(col_end, row_start, text(label_text, color, :right, :top, text_size)); end; end
end

add_borders_and_ticks!(p, (row_start_A, row_end_A), (col_start_A, col_end_A))
add_borders_and_ticks!(p, (1, total_height_p1), (col_start_B, col_end_B))
add_borders_and_ticks!(p, (row_start_A, row_end_A), (col_start_C, col_end_C))
add_sub_matrix_highlight!(p, (row_start_A, row_end_A), (col_start_A, col_end_A), (l3_height, l3_width), "L3")
add_sub_matrix_highlight!(p, (1, total_height_p1), (col_start_B, col_end_B), (l2_height, l2_width), "L2")
add_sub_matrix_highlight!(p, (row_start_A, row_end_A), (col_start_C, col_end_C), (l3_height, l2_width), "")
mul_pos_x1 = col_end_A + padding / 2; eq_pos_x1 = col_end_B + padding / 2; symbol_pos_y1 = total_height_p1 / 2
annotate!(p, mul_pos_x1, symbol_pos_y1, text("*", :white, 24)); annotate!(p, eq_pos_x1, symbol_pos_y1, text("=", :white, 24))

p2_offset = (x_offset_p2, y_offset_p2)
add_borders_and_ticks!(p, (1, scaled_l3_height), (1, scaled_l3_width), color=:cyan, offset=p2_offset, scale=zoom_scale)
add_borders_and_ticks!(p, (1, scaled_l2_height), (z_col_start_B_p2, z_col_end_B_p2), color=:cyan, offset=p2_offset, scale=zoom_scale)
add_borders_and_ticks!(p, (1, scaled_l3_height), (z_col_start_C_p2, z_col_end_C_p2), color=:cyan, offset=p2_offset, scale=zoom_scale)
annotate!(p, x_offset_p2 + scaled_l3_width, y_offset_p2 + scaled_l3_height, text("L3", :cyan, :right, :bottom, 12))
annotate!(p, x_offset_p2 + z_col_end_B_p2, y_offset_p2 + 1, text("L2", :cyan, :right, :top, 12))
add_sub_matrix_highlight!(p, (1,1), (1,1), (scaled_reg_height, scaled_l3_width), "L1", color=:orange, text_size=9, offset=p2_offset)
add_sub_matrix_highlight!(p, (1,1), (z_col_start_B_p2, z_col_end_B_p2), (scaled_l2_height, scaled_reg_width), "L1", color=:orange, text_size=9, offset=p2_offset)
add_sub_matrix_highlight!(p, (1,1), (z_col_start_C_p2, z_col_end_C_p2), (scaled_reg_height, scaled_reg_width), "V0-V31", color=:orange, text_size=5, offset=p2_offset)
mul_pos_x2 = x_offset_p2+scaled_l3_width+scaled_padding/2; eq_pos_x2 = x_offset_p2+z_col_end_B_p2+scaled_padding/2; symbol_pos_y2 = y_offset_p2+zoom_total_height_p2/2
annotate!(p, mul_pos_x2, symbol_pos_y2, text("*", :white, 24)); annotate!(p, eq_pos_x2, symbol_pos_y2, text("=", :white, 24))

p1_A_tl = (col_start_A - 0.5, row_start_A - 0.5); p1_A_tr = (col_start_A + l3_width - 1 + 0.5, row_start_A - 0.5)
p1_A_bl = (col_start_A - 0.5, row_start_A + l3_height - 1 + 0.5); p1_A_br = (col_start_A + l3_width - 1 + 0.5, row_start_A + l3_height - 1 + 0.5)
p1_B_tl = (col_start_B - 0.5, 1 - 0.5); p1_B_tr = (col_start_B + l2_width - 1 + 0.5, 1 - 0.5)
p1_B_bl = (col_start_B - 0.5, l2_height - 1 + 0.5); p1_B_br = (col_start_B + l2_width - 1 + 0.5, l2_height - 1 + 0.5)
p1_C_tl = (col_start_C - 0.5, row_start_A - 0.5); p1_C_tr = (col_start_C + l2_width - 1 + 0.5, row_start_A - 0.5)
p1_C_bl = (col_start_C - 0.5, row_start_A + l3_height - 1 + 0.5); p1_C_br = (col_start_C + l2_width - 1 + 0.5, row_start_A + l3_height - 1 + 0.5)

p2_A_tl = (x_offset_p2 + 1 - 0.5, y_offset_p2 + 1 - 0.5); p2_A_tr = (x_offset_p2 + scaled_l3_width - 1 + 0.5, y_offset_p2 + 1 - 0.5)
p2_A_bl = (x_offset_p2 + 1 - 0.5, y_offset_p2 + scaled_l3_height - 1 + 0.5); p2_A_br = (x_offset_p2 + scaled_l3_width - 1 + 0.5, y_offset_p2 + scaled_l3_height - 1 + 0.5)
p2_B_tl = (x_offset_p2 + z_col_start_B_p2 - 0.5, y_offset_p2 + 1 - 0.5); p2_B_tr = (x_offset_p2 + z_col_end_B_p2 + 0.5, y_offset_p2 + 1 - 0.5)
p2_B_bl = (x_offset_p2 + z_col_start_B_p2 - 0.5, y_offset_p2 + scaled_l2_height - 1 + 0.5); p2_B_br = (x_offset_p2 + z_col_end_B_p2 + 0.5, y_offset_p2 + scaled_l2_height - 1 + 0.5)
p2_C_tl = (x_offset_p2 + z_col_start_C_p2 - 0.5, y_offset_p2 + 1 - 0.5); p2_C_tr = (x_offset_p2 + z_col_end_C_p2 + 0.5, y_offset_p2 + 1 - 0.5)
p2_C_bl = (x_offset_p2 + z_col_start_C_p2 - 0.5, y_offset_p2 + scaled_l3_height - 1 + 0.5); p2_C_br = (x_offset_p2 + z_col_end_C_p2 + 0.5, y_offset_p2 + scaled_l3_height - 1 + 0.5)

for (p1_corner, p2_corner) in [(p1_A_tl, p2_A_tl), (p1_A_tr, p2_A_tr), (p1_A_bl, p2_A_bl), (p1_A_br, p2_A_br), (p1_B_tl, p2_B_tl), (p1_B_tr, p2_B_tr), (p1_B_bl, p2_B_bl), (p1_B_br, p2_B_br), (p1_C_tl, p2_C_tl), (p1_C_tr, p2_C_tr), (p1_C_bl, p2_C_bl), (p1_C_br, p2_C_br)]
    plot!(p, [p1_corner[1], p2_corner[1]], [p1_corner[2], p2_corner[2]], color=:white, style=:dash, label="")
end

savefig(p, "./media/tiling_strategy.png")


