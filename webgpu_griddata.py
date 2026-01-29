import struct
import numpy as np
from scipy.spatial import Delaunay
import wgpu

def pack_params(min_x: float, 
                min_y: float, 
                cell_size_x: float, 
                cell_size_y: float, 
                grid_width: int, 
                grid_height: int, 
                query_count: int, 
                epsilon: float, 
                fill_value: float):
    """
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
    """
    return np.frombuffer(struct.pack(
        '<4f4I4f',
        min_x,
        min_y,
        cell_size_x,
        cell_size_y,
        grid_width,
        grid_height,
        query_count,
        0,  # _pad0
        epsilon,
        fill_value,
        0.0, 0.0  # _pad1
    ), dtype=np.uint32)
    
def build_uniform_grid(points_xy: np.ndarray,
                       triangle_vertex_indices: np.ndarray,
                       grid_width: int,
                       grid_height: int):
    points_xy = np.asarray(points_xy, dtype=np.float64)
    triangle_vertex_indices = np.asarray(triangle_vertex_indices, dtype=np.int64)

    min_x = float(points_xy[:, 0].min())
    max_x = float(points_xy[:, 0].max())
    min_y = float(points_xy[:, 1].min())
    max_y = float(points_xy[:, 1].max())

    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)

    cell_size_x = span_x / grid_width
    cell_size_y = span_y / grid_height

    def get_triangle_grid_bounds(vertex_indices):
        ax, ay = points_xy[vertex_indices[0]]
        bx, by = points_xy[vertex_indices[1]]
        cx, cy = points_xy[vertex_indices[2]]
        
        triangle_min_x = min(ax, bx, cx)
        triangle_max_x = max(ax, bx, cx)
        triangle_min_y = min(ay, by, cy)
        triangle_max_y = max(ay, by, cy)

        x_lower = int(np.floor((triangle_min_x - min_x) / cell_size_x))
        x_upper = int(np.floor((triangle_max_x - min_x) / cell_size_x))
        y_lower = int(np.floor((triangle_min_y - min_y) / cell_size_y))
        y_upper = int(np.floor((triangle_max_y - min_y) / cell_size_y))

        x_lower = max(0, min(grid_width  - 1, x_lower))
        x_upper = max(0, min(grid_width  - 1, x_upper))
        y_lower = max(0, min(grid_height - 1, y_lower))
        y_upper = max(0, min(grid_height - 1, y_upper))
        
        return x_lower, x_upper, y_lower, y_upper
    
    cells_num = grid_width * grid_height
    cell_to_triangles_count = np.zeros(cells_num, dtype=np.uint32)

    # count
    for vertex_indices in triangle_vertex_indices:
        x_lower, x_upper, y_lower, y_upper = get_triangle_grid_bounds(vertex_indices)
        
        for grid_y in range(y_lower, y_upper + 1):
            base = grid_y * grid_width
            for grid_x in range(x_lower, x_upper + 1):
                cell_index = base + grid_x
                cell_to_triangles_count[cell_index] += 1

    cell_triangle_offsets = np.zeros(cells_num + 1, dtype=np.uint32) # last one is total size
    np.cumsum(cell_to_triangles_count, out=cell_triangle_offsets[1:])

    triangle_indices_size = int(cell_triangle_offsets[-1])
    cell_triangle_indices = np.empty(triangle_indices_size, dtype=np.uint32)
    cell_to_write_pos = cell_triangle_offsets[:-1].copy()

    # fill
    for triangle_index, vertex_indices in enumerate(triangle_vertex_indices):
        x_lower, x_upper, y_lower, y_upper = get_triangle_grid_bounds(vertex_indices)
        
        for grid_y in range(y_lower, y_upper + 1):
            base = grid_y * grid_width
            for grid_x in range(x_lower, x_upper + 1):
                cell_index = base + grid_x
                cell_triangle_indices[cell_to_write_pos[cell_index]] = np.uint32(triangle_index)
                cell_to_write_pos[cell_index] += 1

    return (min_x, min_y, cell_size_x, cell_size_y, cell_triangle_offsets, cell_triangle_indices)

class GriddataLinearWebGPU:
    """
    One-time setup:
      - Delaunay triangulation (CPU)
      - Uniform grid CSR index (CPU)
      - WebGPU pipeline + static buffers (GPU)
    Then:
      - query(query_points_xy) many times, updating only params + query buffer.
    """

    def __init__(self,
                 points_xy: np.ndarray,
                 point_values: np.ndarray,
                 *,
                 fill_value=np.nan,
                 epsilon: float = 1e-10,
                 grid_width: int = 512,
                 grid_height: int = 512,
                 workgroup_size: int = 256,
                 shader_path: str = "griddata_linear_uniform_grid.wgsl"):

        points_xy = np.asarray(points_xy)
        point_values = np.asarray(point_values)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy must be (N,2)")
        if point_values.ndim != 1 or point_values.shape[0] != points_xy.shape[0]:
            raise ValueError("point_values must be (N,) and match points_xy")

        self.fill_value = float(fill_value)
        self.epsilon = float(epsilon)
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.workgroup_size = int(workgroup_size)

        delaunay = Delaunay(points_xy)
        tri_vidx_2d = delaunay.simplices.astype(np.uint32)  # (T,3)

        min_x, min_y, csx, csy, offsets, indices = build_uniform_grid(
            points_xy, tri_vidx_2d, self.grid_width, self.grid_height
        )

        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.cell_size_x = float(csx)
        self.cell_size_y = float(csy)

        self.points_xy_flat = np.ascontiguousarray(points_xy.astype(np.float32).ravel())
        self.point_values_flat = np.ascontiguousarray(point_values.astype(np.float32).ravel())
        self.tri_vidx_flat = np.ascontiguousarray(tri_vidx_2d.ravel().astype(np.uint32))
        self.offsets_flat = np.ascontiguousarray(offsets.astype(np.uint32).ravel())
        self.indices_flat = np.ascontiguousarray(indices.astype(np.uint32).ravel())

        with open(shader_path, "r", encoding="utf-8") as f:
            shader_code = f.read()
        shader_code = shader_code.replace("WORKGROUP_SIZE", str(self.workgroup_size))

        self.device = wgpu.utils.get_default_device()
        self.queue = self.device.queue

        self.shader = self.device.create_shader_module(code=shader_code)

        bind_group_layout_entries = []
        bind_group_layout_entries.append({
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": { "type": wgpu.BufferBindingType.uniform }
            })
        for i in range(1, 7):
            bind_group_layout_entries.append({
                "binding": i,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": { "type": wgpu.BufferBindingType.read_only_storage }
            })
        bind_group_layout_entries.append({
            "binding": 7,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": { "type": wgpu.BufferBindingType.storage }
        })

        self.bind_group_layout = self.device.create_bind_group_layout(entries=bind_group_layout_entries)
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])
        self.pipeline = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": self.shader, "entry_point": "main"},
        )

        # dynamic params buffer
        self.buf_params = self.device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        
        # static buffers
        self.buf_points_xy = self.device.create_buffer_with_data(data=self.points_xy_flat, usage=wgpu.BufferUsage.STORAGE)
        self.buf_point_values = self.device.create_buffer_with_data(data=self.point_values_flat, usage=wgpu.BufferUsage.STORAGE)
        self.buf_tri_vidx = self.device.create_buffer_with_data(data=self.tri_vidx_flat, usage=wgpu.BufferUsage.STORAGE)
        self.buf_offsets = self.device.create_buffer_with_data(data=self.offsets_flat, usage=wgpu.BufferUsage.STORAGE)
        self.buf_indices = self.device.create_buffer_with_data(data=self.indices_flat, usage=wgpu.BufferUsage.STORAGE)

        # dynamic buffers allocated per query size
        self.query_count = 0
        self.buf_query_xy = None
        self.buf_output = None
        self.buf_readback = None
        self.bind_group = None

    def get_device_description(self) -> str:
        info = self.device.adapter.info
        
        device_name = info.get("device", "Unknown Device")
        vendor = info.get("vendor", "Unknown Vendor")
        architecture = info.get("architecture", "Unknown Architecture")
        device_type = info.get("device_type", "Unknown Type")
        backend = info.get("backend", "Unknown Backend")
        
        return f"{vendor} {device_name} ({device_type}, {architecture}, backend: {backend})"

    def _ensure_query_buffers(self, query_count: int, element_size: int):
        # reallocate only if size changed
        query_count = int(query_count)
        if query_count == self.query_count and self.bind_group is not None:
            return
        self.query_count = query_count

        self.buf_query_xy = self.device.create_buffer(
            size=2 * query_count * element_size, 
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.buf_output = self.device.create_buffer(
            size=query_count * element_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        self.buf_readback = self.device.create_buffer(
            size=query_count * element_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        entries = []
        for i, buffer in enumerate([
            self.buf_params,
            self.buf_points_xy,
            self.buf_point_values,
            self.buf_query_xy,
            self.buf_tri_vidx,
            self.buf_offsets,
            self.buf_indices,
            self.buf_output,
        ]):
            entries.append({"binding": i, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}})
    
        self.bind_group = self.device.create_bind_group(layout=self.bind_group_layout, entries=entries)

    def query(self, query_points_xy: np.ndarray) -> np.ndarray:
        query_points_xy = np.asarray(query_points_xy)
        if query_points_xy.ndim != 2 or query_points_xy.shape[1] != 2:
            raise ValueError("query_points_xy must be (M,2)")

        FLOAT_SIZE = 4
        query_count = int(query_points_xy.shape[0])
        self._ensure_query_buffers(query_count, element_size=FLOAT_SIZE)

        params_u32 = pack_params(
            self.min_x, self.min_y,
            self.cell_size_x, self.cell_size_y,
            self.grid_width, self.grid_height,
            query_count,
            self.epsilon, self.fill_value
        )

        query_flat = np.ascontiguousarray(query_points_xy.astype(np.float32).ravel())

        # upload dynamic inputs
        self.queue.write_buffer(self.buf_params, 0, params_u32.tobytes())
        self.queue.write_buffer(self.buf_query_xy, 0, query_flat.tobytes())

        dispatch_x = (query_count + self.workgroup_size - 1) // self.workgroup_size

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, self.bind_group)
        cp.dispatch_workgroups(dispatch_x, 1, 1)
        cp.end()
        
        enc.copy_buffer_to_buffer(self.buf_output, 0, self.buf_readback, 0, query_count * FLOAT_SIZE)
        self.queue.submit([enc.finish()])

        # readback
        self.buf_readback.map_sync(wgpu.MapMode.READ)
        data = self.buf_readback.read_mapped(0, query_count * FLOAT_SIZE, copy=True)
        self.buf_readback.unmap()
        
        result = np.frombuffer(data, dtype=np.float32)
        
        return result

def webgpu_griddata_linear_2d(points_xy: np.ndarray,
                          point_values: np.ndarray,
                          query_points_xy: np.ndarray,
                          fill_value=np.nan,
                          *,
                          grid_width=512,
                          grid_height=512,
                          epsilon=1e-10,
                          workgroup_size=256) -> np.ndarray:
    """
    Perform 2D linear interpolation on scattered data points using Delaunay triangulation.

    Parameters:
    - points_xy: (n, 2) array of input points.
    - point_values: (n,) array of values at the input points.
    - query_points_xy: (m, 2) array of points where interpolation is to be performed.
    - fill_value: value to use for points outside the convex hull.

    Returns:
    - interpolated_values: (m,) array of interpolated values at query_points_xy.
    """
    runner = GriddataLinearWebGPU(
        points_xy, point_values,
        fill_value=fill_value,
        epsilon=epsilon,
        grid_width=grid_width,
        grid_height=grid_height,
        workgroup_size=workgroup_size,
    )
    return runner.query(query_points_xy)
