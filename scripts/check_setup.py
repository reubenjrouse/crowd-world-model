"""Check that every tool the project needs is installed and working.

Run:  python scripts/check_setup.py
Each check prints OK or FAIL with a short reason.
"""

import importlib
import time

failures = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  OK    {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:  # noqa: BLE001 - we want to report any failure
        failures.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def version_of(module):
    return lambda: importlib.import_module(module).__version__


print("1. Libraries installed")
for mod in ["numpy", "scipy", "pandas", "pyarrow", "h5py", "shapely",
            "matplotlib", "jupedsim", "pedpy", "torch", "crowdwm"]:
    check(mod, version_of(mod))


print("\n2. Graphics card (GPU)")


def gpu_available():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch can't see the GPU (CPU-only build or driver issue)")
    props = torch.cuda.get_device_properties(0)
    return f"{props.name}, {props.total_memory / 1e9:.1f} GB, CUDA {torch.version.cuda}"


def gpu_math():
    # Multiply two big grids of numbers in half precision (what training will use)
    import torch
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12
    return f"{tflops:.1f} TFLOPS half-precision"


check("GPU visible to PyTorch", gpu_available)
check("GPU does math", gpu_math)


print("\n3. Crowd simulator (JuPedSim)")


def tiny_simulation():
    # 10 people in a 10 m x 10 m room walk to a 2 m wide exit on the right wall
    import jupedsim as jps
    from shapely import Polygon

    room = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModel(), geometry=room)
    exit_id = sim.add_exit_stage([(9.5, 4), (10, 4), (10, 6), (9.5, 6)])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))
    for i in range(10):
        sim.add_agent(jps.CollisionFreeSpeedModelAgentParameters(
            journey_id=journey_id, stage_id=exit_id,
            position=(1 + (i % 5) * 1.0, 3 + (i // 5) * 4.0)))

    t0 = time.perf_counter()
    while sim.agent_count() > 0 and sim.iteration_count() < 10_000:
        sim.iterate()
    if sim.agent_count() > 0:
        raise RuntimeError(f"{sim.agent_count()} people never reached the exit")
    return (f"all 10 left after {sim.elapsed_time():.1f} s simulated, "
            f"computed in {time.perf_counter() - t0:.2f} s")


check("tiny simulation", tiny_simulation)


print("\n4. Real-data loader (PedPy)")


def pedpy_geometry():
    import pedpy
    area = pedpy.WalkableArea([(0, 0), (10, 0), (10, 10), (0, 10)])
    return f"walkable area {area.polygon.area:.0f} m^2"


check("PedPy geometry", pedpy_geometry)


print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    raise SystemExit(1)
print("Everything works.")
