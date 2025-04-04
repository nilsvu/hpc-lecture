# Parallelization

First: make sure your serial program is reasonably fast:

- Common sense: Don't repeat expensive computations. Avoid copying memory around.
- Python: Avoid loops. Use Numpy.
- Check which functions are expensive and fix those first ("profiling"):
  ```
  python -m cProfile -s time YOUR_SCRIPT.py
  ```

## Example 1: Ray tracing around a black hole

- "Embarrassingly parallel": each ray is independent
- Quick Python parallelization: use `multiprocessing.Pool`

## Example 2: Gravitational N-body simulation

- Every particle interacts with every other: cost scales with N^2
- Python implementation (no parallelization):
    - 1e3 particles for 200 time steps on Apple M2 Pro chip: 13 minutes
- Quick Python parallelization: use `numba.jit` ("just-in-time" compilation to C)
    - 1e3 particles, single core: 54 seconds
    - 1e3 particles, 12 cores: 6 seconds. 1e4 particles: 570 seconds.
- Next step: OpenMP shared-memory parallelization in C++
    - 1e3 particles, single thread: 0.5 seconds
    - 1e3 particles, 12 threads: 0.1 seconds
    - 1e4 particles, 12 threads: 6.7 seconds
- Even better: Kokkos parallelization can also run on GPU
    - 1e4 particles, 12 threads (OpenMP on Apple M2 Pro): 7.5 seconds
    - 1e4 particles, NVIDIA A100 GPU: 1.1 seconds
