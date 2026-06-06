# openfoam-metal-solver — Developer Context

## What this is
A hybrid PCG linear solver plugin for OpenFOAM 11 on macOS Apple Silicon. Registered via
OpenFOAM's runtime type selection (RTS) as `metalPCG` (solver) and `MetalGCGS` (smoother).
Plugin loads as a wmake shared library — no fork of OpenFOAM source required.

## Key constraints
- Metal Apple Silicon has no float64. GPU kernels run float32; CPU handles all float64.
- Bit-for-bit identical output vs vanilla PCG is not a goal — same convergence within tolerance is.
- Plugin loads alongside production build at `/Volumes/OpenFOAM/OpenFOAM-11`.
- Serial only for full GPU benefit — see parallelism section below.

## Directory structure
```
src/metalPCGSolver/
  MetalPCGSolver.H       — lduMatrix::solver subclass, registered as metalPCG
  MetalPCGSolver.C       — solve(): hybrid GPU PCG loop / CPU fallback with hybridAmul
  MetalGCGSSmoother.H    — lduMatrix::smoother subclass, registered as MetalGCGS
  MetalGCGSSmoother.C    — smooth(): GPU GCGS (serial) / CPU GS delegate (parallel)
  metalSpMV.h            — Metal backend public interface
  metalSpMV.mm           — Obj-C++: MetalShared singleton, kernels, PCG loop, GCGS
  Make/
    files                — wmake source list
    options              — link flags (-framework Metal -framework Foundation)
tutorials/cavity-metal/  — 100K cell cavity case
tutorials/benchmark-1m-gamg/ — 1M cell benchmark case (metalPCG + GAMG + MetalGCGS)
tests/residual_check.py  — compare residual history vs vanilla PCG
```

## Architecture

### metalSpMV.mm structure
- `kShaderSrc` — embedded Metal shader source string, compiled at runtime via `newLibraryWithSource`
- `MetalShared` — process-wide singleton: `MTLDevice`, `MTLCommandQueue`, pipeline states
- `MetalSpMV::Impl` — per-instance: CSR buffers, PCG buffers, GCGS color buffers
- `MetalSpMV::setup()` — uploads CSR (float32), fingerprint-cached on (nRows, nnz, valSum)
- `MetalSpMV::setupGCGS()` — greedy graph coloring, uploads per-color row index buffers
- `MetalSpMV::solveGPUPCG()` — 3 CB per iteration: dot(z,r) → p-update+SpMV+dot(Ap,p) → SAXPY+norm
- `MetalSpMV::smoothGCGS()` — forward+backward color sweep, float→double round-trip per call

### Metal kernels (in kShaderSrc)
| Kernel | Purpose |
|--------|---------|
| `sparseMV` | CSR SpMV, float32 |
| `dot_partial` | Partial reduction dot product (256-thread threadgroups) |
| `sumabs_partial` | Partial reduction of \|a[i]\| for residual norm |
| `saxpy` | y += alpha * x |
| `saxpy_sub` | y -= alpha * x |
| `pcg_p_update` | p = z + beta * p |
| `gs_color_sweep` | GS update for one color: x[i] = (b[i] - off-diag) / diag[i] |

### MetalPCGSolver.C — solve() flow
1. Build CSR from LDU addressing, upload via `sparseMV_->setup()`
2. Define `hybridAmul` lambda:
   - Serial+GPU: `sparseMV_->multiply()` only
   - Parallel+GPU: `initMatrixInterfaces` → `sparseMV_->multiply()` → `updateMatrixInterfaces`
   - No GPU: `matrix_.Amul()`
3. Compute initial residual and normFactor via `hybridAmul`
4. Serial+GPU path: `sparseMV_->solveGPUPCG()` with preconditioner lambda (CPU)
5. CPU fallback path (parallel or GPU failure): standard PCG loop using `hybridAmul`

### MetalGCGSSmoother.C — smooth() flow
1. Check `levelHasInterfaces` (processor interface = parallel run)
2. Serial+GPU+GCGS ready → `sparseMV_->smoothGCGS()` (forward+backward sweep)
3. Parallel or no GPU → `GaussSeidelSmoother::smooth()` (CPU, preserves GAMG convergence)
4. Coarse level, serial, no GPU → damped Jacobi fallback (omega=2/3)

## Benchmark results (1M cells, 50 steps, Apple Silicon)

### Serial (1 MPI rank)

| Solver | Preconditioner | Smoother | Time |
|--------|---------------|----------|------|
| PCG | DIC | — | 253.2s |
| metalPCG v1 (GPU SpMV only) | DIC | — | 188.7s |
| metalPCG v2 (full GPU loop) | DIC | — | 173.4s |
| PCG | GAMG | GaussSeidel | 58.3s |
| metalPCG v1 | GAMG | GaussSeidel | 59.6s |
| metalPCG v2 | GAMG | GaussSeidel | 59.0s |
| metalPCG v2 | GAMG | MetalGCGS | **50.5s** |

### Parallel (8 MPI ranks, scotch decomposition)

| Solver | Smoother | Time |
|--------|----------|------|
| PCG (CPU) | GaussSeidel | 14.2s |
| metalPCG | MetalGCGS (CPU GS fallback) | ~14–15s |

## Key findings

**GAMG is the dominant win.** DIC→GAMG: 253s→58s (4×). GPU SpMV on top: 58s→50s (15%).
Always use GAMG before thinking about GPU acceleration.

**GPU SpMV scales with iteration count.** With DIC (~220 iters/step): +32%. With GAMG (1–9
iters/step): SpMV is a small fraction of V-cycle time — diminishing returns.

**Jacobi smoothing killed GAMG convergence.** Damped Jacobi GPU smoother ran at 107s (worse
than vanilla). Root cause: 1 Jacobi sweep ≈ 0.25 GS sweeps in smoothing quality → 3–5× more
outer PCG iterations. Float32 precision was NOT the issue — Jacobi DID converge to 1e-6.
Graph-colored GaussSeidel (GCGS) restores GS quality while being GPU-parallelisable.

**MPI parallelism beats single-rank GPU.** 8-rank CPU: 14.2s vs 1-rank GPU: 50.5s. The CPU
performance cores on Apple Silicon are fast; MPI scaling is straightforward.

**GPU doesn't help in parallel.** 8 MPI ranks = 8 separate OS processes. Metal has no
multi-process concurrent GPU execution (unlike CUDA MPS). The 8 MTLCommandQueues serialize
on one physical GPU — contention cancels per-rank SpMV speedup. Result: ~14–15s, same as CPU.

**Unified memory limits GPU advantage.** CPU and GPU share DRAM on Apple Silicon. SpMV is
bandwidth-bound; without a bandwidth cliff (as exists between discrete GPU HBM and CPU DDR),
the GPU advantage is smaller than on NVIDIA hardware.

## Development history

| Milestone | What was built | Key result |
|-----------|---------------|------------|
| 1 | GPU SpMV via Metal MPSSparseMatrix, CSR upload | +16% at 100K cells, +25% at 1M cells vs vanilla PCG+DIC |
| 2 | GAMG preconditioner benchmarked | GAMG alone gives 4× over DIC; GPU SpMV with GAMG ≈ CPU |
| 3a | Full GPU PCG loop (dot + SAXPY + SpMV all in Metal) | +32% vs CPU on DIC path |
| 3b | Damped Jacobi GPU smoother | 107s — abandoned; poor smoothing quality, not float32 |
| 4 | Graph-colored GaussSeidel (MetalGCGS) | 50.5s, +15% vs CPU GAMG baseline |
| 5 | MPI parallel support (hybrid Amul + GS fallback) | ~14–15s, matches CPU parallel |
