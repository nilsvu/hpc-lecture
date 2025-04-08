# Parallelization

First: make sure your serial program is reasonably fast:

- Common sense: Don't repeat expensive computations. Avoid copying memory around.
- Python: Avoid loops. Use Numpy.
- Check which functions are expensive and fix those first ("profiling"):
  ```
  python -m cProfile -s time YOUR_SCRIPT.py
  ```
- Compile C++ with optimizations: `-O3` flag or `cmake -D CMAKE_BUILD_TYPE=Release`

## Example 1: Ray tracing around a black hole

- "Embarrassingly parallel": each ray is independent
- Quick Python parallelization: use `multiprocessing.Pool`

## Example 2: Gravitational N-body simulation

- Every particle interacts with every other: cost scales with N^2
- Python implementation (no parallelization):
    - 1e3 particles for 200 time steps on Apple M2 Pro chip: **13 minutes**
- Quick Python parallelization: use `numba.jit` ("just-in-time" compilation to C)
    - 1e3 particles, single core: 54 seconds
    - 1e3 particles, 12 cores: **6 seconds**. 1e4 particles: 570 seconds.
- Next step: OpenMP shared-memory parallelization in C++
    - 1e3 particles, single thread: 0.5 seconds
    - 1e3 particles, 12 threads: **0.1 seconds**
    - 1e4 particles, 12 threads: **6.7 seconds**
- Even better: Kokkos parallelization can also run on GPU
    - 1e4 particles, 12 threads (OpenMP on Apple M2 Pro): 7.5 seconds
    - 1e4 particles, NVIDIA A100 GPU: **1.1 seconds**

## Example 3: Thermal diffusion (PDE)

- Domain decomposition with nearest-neighbor communication
- Distributed on multiple nodes (MPI), each with shared-memory parallelization
  (Numba/OpenMP)
- Launch 1 process per node or 1 process per NUMA domain / socket, then spawn as
  many threads as cores available:
  ```sh
  #!/bin/bash
  #SBATCH --job-name=<JOB_NAME>
  #SBATCH --nodes=<NUM_NODES>
  #SBATCH --ntasks-per-node=<NUM_SOCKETS>
  #SBATCH --cpus-per-task=<NUM_CORES_PER_SOCKET>
  #SBATCH --time=00:30:00
  
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export NUMBA_NUM_THREADS=$OMP_NUM_THREADS
  
  srun python <YOUR_SCRIPT.py>
  ```
- Check CPU affinity and thread placement to make sure that indeed each
  NUMA domain runs 1 process, which places one thread on each of its cores.
  Check supercomputer documentation for advice. You can experiment with
  different configurations (e.g. hardware multithreading).
