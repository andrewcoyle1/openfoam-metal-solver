# openfoam-metal-solver — Project Goals

## What this is
A hybrid PCG linear solver plugin for OpenFOAM 11 on macOS Apple Silicon. Registered via OpenFOAM's runtime type selection (RTS) as `metalPCG`. Users set it in `fvSolution` exactly like any built-in solver.

## Architecture
- **Preconditioner:** ILU(0) on CPU — uses OpenFOAM's existing DILU
- **SpMV:** Sparse matrix-vector multiply on GPU via Metal `MPSSparseMatrix`
- **Everything else** (dot products, vector updates, convergence checks) on CPU
- **Packaging:** wmake shared library plugin; no fork of OpenFOAM source required

## Milestone 1 targets

| Target | Value |
|--------|-------|
| Completion bar | Tutorial case runs to completion using `metalPCG` in fvSolution |
| Mesh size for benchmarking | ~100K cells |
| Fields supported | All scalar and vector fields (p, U, T, k, ε, …) |
| Parallel | Serial only (1 MPI rank) |
| Correctness | Converges to same residual tolerance as vanilla PCG in ≤ same iterations |
| Precision | Float32 on GPU (Metal has no Float64 hardware on Apple Silicon) |

## Milestone 1 benchmark results

| Solver | ExecutionTime (100 steps, 100K cells, laminar cavity) |
|--------|------------------------------------------------------|
| PCG (vanilla, 100K) | 23.3s |
| metalPCG (100K, final) | **19.5s** (+16%) |
| PCG (vanilla, 1M) | 253.2s |
| metalPCG (1M) | **188.7s** (+25%) |

Speedup grows with mesh size as GPU memory bandwidth advantage over CPU becomes dominant relative to Metal command buffer dispatch overhead. Correctness confirmed at both sizes.

## Milestone 2 benchmark results — GAMG preconditioner (1M cells, 50 steps)

| Solver | Preconditioner | Iters/step | ExecutionTime |
|--------|---------------|------------|---------------|
| PCG (CPU) | DIC | ~220 | 253.2s |
| metalPCG (GPU SpMV) | DIC | ~220 | 188.7s |
| PCG (CPU) | GAMG | 1–9 | **58.3s** |
| metalPCG (GPU SpMV) | GAMG | 1–9 | 59.6s |

Key finding: GAMG preconditioner is the dominant win (4–5× over DIC) regardless of SpMV backend.
With GAMG, PCG+GAMG ≈ metalPCG+GAMG because GAMG V-cycle smoothers (GaussSeidel) are
CPU-bound with sequential data dependencies — GPU SpMV is no longer the bottleneck.

GPU SpMV acceleration is most valuable with DIC (many iterations). To make GPU matter with
GAMG, the GaussSeidel smoother would need to run on GPU — harder due to triangular solve.

## Milestone 2 follow-on items
- GPU-accelerated GaussSeidel smoother for GAMG: colouring or polynomial approximation
- Batch command buffers for DIC path to further reduce dispatch overhead
- Multi-rank parallel: each MPI rank gets its own `MTLCommandQueue` on the shared device

## Key constraints
- Metal shaders on Apple Silicon have no Float64. SpMV runs in Float32; CPU handles all Float64 arithmetic.
- Bit-for-bit identical output vs vanilla PCG is not a goal — same convergence within tolerance is.
- Plugin must load cleanly alongside the production build at `/Volumes/OpenFOAM/OpenFOAM-11`.

## Directory structure
```
src/metalPCGSolver/
  MetalPCGSolver.H       — lduMatrix::solver subclass
  MetalPCGSolver.C       — solve() loop: DILU on CPU, SpMV on GPU
  metalSpMV.h            — Metal wrapper interface (plain C++ header)
  metalSpMV.mm           — Obj-C++ Metal implementation
  Make/
    files                — wmake source list
    options              — link flags (-framework Metal -framework MetalPerformanceShaders)
tutorials/cavity-metal/  — 100K cell cavity case with metalPCG configured
tests/
  residual_check.py      — compare residual history vs vanilla PCG
```
