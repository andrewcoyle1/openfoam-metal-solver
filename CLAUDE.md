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
| PCG (vanilla) | 23.3s |
| metalPCG v1 (re-upload matrix every step) | 20.3s |
| metalPCG v2 (matrix fingerprint cache) | 20.2s |
| metalPCG v3 (matrix + vector buffer cache) | **19.5s** |

**16% faster** than vanilla PCG at 100K cells. Correctness confirmed: residuals and iteration counts match to float32 precision. The gain is primarily from GPU SpMV throughput, partially offset by Metal command buffer dispatch overhead (~hundreds of µs per call) which dominates at small problem sizes.

## Milestone 2 (follow-on, not in scope yet)
- Batch multiple SpMV calls into a single command buffer to reduce dispatch overhead
- Benchmark at 1M+ cells where GPU throughput dominates over dispatch overhead
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
