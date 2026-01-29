struct Params {
    min_x: f32,
    min_y: f32,
    cell_size_x: f32,
    cell_size_y: f32,
    grid_width: u32,
    grid_height: u32,
    query_count: u32,
    _pad0: u32,
    epsilon: f32,
    fill_value: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> points_xy : array<f32>;
@group(0) @binding(2) var<storage, read> point_values : array<f32>;
@group(0) @binding(3) var<storage, read> query_points_xy : array<f32>;
@group(0) @binding(4) var<storage, read> triangle_vertex_indices : array<u32>;
@group(0) @binding(5) var<storage, read> cell_triangle_offsets : array<u32>;
@group(0) @binding(6) var<storage, read> cell_triangle_indices : array<u32>;
@group(0) @binding(7) var<storage, read_write> output_values : array<f32>;

fn load_point_xy(point_index: u32) -> vec2<f32> {
    return vec2<f32>(points_xy[2u * point_index + 0u], points_xy[2u * point_index + 1u]);
}

fn signed_area(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

fn barycentric_coords(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let enclosing_triangle_area = signed_area(a, b, c);
    let w0 = signed_area(b, c, p) / enclosing_triangle_area;
    let w1 = signed_area(c, a, p) / enclosing_triangle_area;
    let w2 = signed_area(a, b, p) / enclosing_triangle_area;
    return vec3<f32>(w0, w1, w2);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_index = gid.x;
    if (query_index >= params.query_count) { 
        return; 
    }

    let query_xy = vec2<f32>(query_points_xy[2u * query_index + 0u], query_points_xy[2u * query_index + 1u]);

    let cell_x = i32(floor((query_xy.x - params.min_x) / params.cell_size_x));
    let cell_y = i32(floor((query_xy.y - params.min_y) / params.cell_size_y));
    if (cell_x < 0 || u32(cell_x) >= params.grid_width || cell_y < 0 || u32(cell_y) >= params.grid_height) {
        output_values[query_index] = params.fill_value;
        return;
    }

    let cell_index = u32(cell_y) * params.grid_width + u32(cell_x);
    let tri_begin = cell_triangle_offsets[cell_index];
    let tri_end = cell_triangle_offsets[cell_index + 1u];

    var result_value = params.fill_value;

    for (var tri_cursor = tri_begin; tri_cursor < tri_end; tri_cursor++) {
        let tri_index = cell_triangle_indices[tri_cursor];
        let tri_base = 3u * tri_index;

        let vertex_index_0 = triangle_vertex_indices[tri_base + 0u];
        let vertex_index_1 = triangle_vertex_indices[tri_base + 1u];
        let vertex_index_2 = triangle_vertex_indices[tri_base + 2u];

        let v0 = load_point_xy(vertex_index_0);
        let v1 = load_point_xy(vertex_index_1);
        let v2 = load_point_xy(vertex_index_2);

        if (abs(signed_area(v0, v1, v2)) < 1e-20) { continue; }

        let bary = barycentric_coords(query_xy, v0, v1, v2);

        if (bary.x >= -params.epsilon && bary.y >= -params.epsilon && bary.z >= -params.epsilon) {
            result_value = bary.x * point_values[vertex_index_0] + bary.y * point_values[vertex_index_1] + bary.z * point_values[vertex_index_2];
            break;
        }
    }

    output_values[query_index] = result_value;
}
