import time
import numpy as np
from webgpu_griddata import webgpu_griddata_linear_2d, GriddataLinearWebGPU
from scipy.interpolate import griddata as scipy_griddata


def compare_with_scipy(points_xy, point_values, query_points_xy, *,
                       fill_value=np.nan,
                       grid_width=512, grid_height=512,
                       warmup=2, iters=10):
    points_xy = np.asarray(points_xy)
    point_values = np.asarray(point_values)
    query_points_xy = np.asarray(query_points_xy)

    # correctness
    ref = scipy_griddata(points_xy, point_values, query_points_xy,
                         method="linear", fill_value=fill_value)
    gpu = webgpu_griddata_linear_2d(points_xy, point_values, query_points_xy,
                                fill_value=fill_value, grid_width=grid_width, grid_height=grid_height)

    ref = np.asarray(ref, dtype=np.float64)
    gpu = np.asarray(gpu, dtype=np.float64)

    mask = np.isfinite(ref)
    if np.any(mask):
        diff = np.abs(ref[mask] - gpu[mask])
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff * diff)))
    else:
        max_abs = float("nan")
        mean_abs = float("nan")
        rmse = float("nan")

    def time_avg(fn):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        return (t1 - t0) / iters

    t_scipy = time_avg(lambda: scipy_griddata(points_xy, point_values, query_points_xy,
                                              method="linear", fill_value=fill_value))
    
    # Measure GPU full_pipeline queries (setup + query each time)
    setup_times = []
    query_times = []
    full_times = []
    
    def gpu_full_pipeline_query():
        t_start = time.perf_counter()
        runner = GriddataLinearWebGPU(
            points_xy, point_values,
            fill_value=fill_value,
            grid_width=grid_width,
            grid_height=grid_height
        )
        t_setup = time.perf_counter()
        result = runner.query(query_points_xy)
        t_end = time.perf_counter()
        
        setup_times.append(t_setup - t_start)
        query_times.append(t_end - t_setup)
        full_times.append(t_end - t_start)
        return result
    
    for _ in range(warmup):
        gpu_full_pipeline_query()

    setup_times.clear()
    query_times.clear()
    full_times.clear()
    
    for _ in range(iters):
        gpu_full_pipeline_query()
    
    t_gpu_full_pipeline_time = np.mean(full_times)
    t_gpu_full_pipeline_setup = np.mean(setup_times)
    t_gpu_full_pipeline_query = np.mean(query_times)
    
    # Measure reusable instance (setup once, query many times)
    runner_reusable = GriddataLinearWebGPU(
        points_xy, point_values,
        fill_value=fill_value,
        grid_width=grid_width,
        grid_height=grid_height
    )
    t_gpu_query_reusable = time_avg(lambda: runner_reusable.query(query_points_xy))

    return {
        "finite_ref_fraction": float(np.mean(mask)),
        "max_abs_diff_on_finite": max_abs,
        "mean_abs_diff_on_finite": mean_abs,
        "rmse_on_finite": rmse,
        "scipy_seconds_per_call": float(t_scipy),
        "gpu_full_pipeline_time": float(t_gpu_full_pipeline_time),
        "gpu_full_pipeline_setup_time": float(t_gpu_full_pipeline_setup),
        "gpu_full_pipeline_query_time": float(t_gpu_full_pipeline_query),
        "gpu_query_reusable_time": float(t_gpu_query_reusable),
        "speedup_scipy_over_gpu_full_pipeline": float(t_scipy / t_gpu_full_pipeline_time) if t_gpu_full_pipeline_time > 0 else float("inf"),
        "speedup_scipy_over_gpu_query_reusable": float(t_scipy / t_gpu_query_reusable) if t_gpu_query_reusable > 0 else float("inf"),
        "device_description": runner_reusable.get_device_description(),
    }


def make_test_data(*,
                   n_points,
                   n_query,
                   seed=0):
    rng = np.random.default_rng(seed)

    points_xy = rng.random((n_points, 2), dtype=np.float64)

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    values = (np.sin(10.0 * x) + np.cos(7.0 * y) + 0.1 * np.sin(25.0 * (x + y))).astype(np.float64)

    query_points_xy = rng.random((n_query, 2), dtype=np.float64)

    return points_xy, values, query_points_xy


def pretty_time(x: float) -> str:
    if x < 1e-6:
        return f"{x * 1e9:.2f} ns"
    if x < 1e-3:
        return f"{x * 1e6:.2f} µs"
    if x < 1:
        return f"{x * 1e3:.2f} ms"
    return f"{x:.3f} s"


def main():
    n_points = 20000
    n_query = 500000

    grid_width = 512
    grid_height = 512

    fill_value = np.nan

    print("Generating data...")
    points_xy, values, query_xy = make_test_data(
        n_points=n_points,
        n_query=n_query,
        seed=0,
    )

    print(f"points: {points_xy.shape}, query: {query_xy.shape}, grid: {grid_width}x{grid_height}")
    print("Running comparison...")

    stats = compare_with_scipy(
        points_xy, values, query_xy,
        fill_value=fill_value,
        grid_width=grid_width, grid_height=grid_height,
        warmup=2, iters=5,
    )

    print("\n=== GPU Device ===")
    print(f"{stats['device_description']}")
    
    print("\n=== Correctness (vs SciPy griddata linear) ===")
    print(f"Finite ref fraction: {stats['finite_ref_fraction'] * 100:.2f}%")
    print(f"Max abs diff (finite only):  {stats['max_abs_diff_on_finite']:.6g}")
    print(f"Mean abs diff (finite only): {stats['mean_abs_diff_on_finite']:.6g}")
    print(f"RMSE (finite only):          {stats['rmse_on_finite']:.6g}")

    print("\n=== Speed ===")
    print(f"SciPy: {pretty_time(stats['scipy_seconds_per_call'])} per call")
    print(f"\nGPU (full pipeline - setup + query each time):")
    print(f"  Setup: {pretty_time(stats['gpu_full_pipeline_setup_time'])} per call")
    print(f"  Query: {pretty_time(stats['gpu_full_pipeline_query_time'])} per call")
    print(f"  Total: {pretty_time(stats['gpu_full_pipeline_time'])} per call")
    print(f"  Speedup vs SciPy: {stats['speedup_scipy_over_gpu_full_pipeline']:.2f}x")
    print(f"\nGPU (reusable instance - setup once, query many times):")
    print(f"  Query:  {pretty_time(stats['gpu_query_reusable_time'])} per call")
    print(f"  Speedup vs SciPy: {stats['speedup_scipy_over_gpu_query_reusable']:.2f}x")

    tol_max = 5e-3
    if np.isfinite(stats["max_abs_diff_on_finite"]) and stats["max_abs_diff_on_finite"] > tol_max:
        raise SystemExit(f"\nFAIL: max_abs_diff {stats['max_abs_diff_on_finite']:.6g} > {tol_max}\n")

    print("\nOK")


if __name__ == "__main__":
    main()
