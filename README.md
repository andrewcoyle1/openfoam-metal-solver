# openfoam-metal-solver

A GPU-accelerated linear solver plugin for OpenFOAM 11 on macOS Apple Silicon, using Apple's Metal framework. Implemented as a wmake shared library — no fork of OpenFOAM required.

Provides two registered types:

- **`metalPCG`** — drop-in replacement for OpenFOAM's `PCG` solver
- **`MetalGCGS`** — drop-in replacement for the `GaussSeidel` GAMG smoother

## Requirements

- macOS 13+ on Apple Silicon (M1/M2/M3)
- OpenFOAM 11 installed at `/Volumes/OpenFOAM/OpenFOAM-11`
- Xcode command line tools (for Metal shader compilation at runtime)

## Build

```bash
source /Volumes/OpenFOAM/OpenFOAM-11/etc/bashrc
cd src/metalPCGSolver
wmake
```

Produces `$FOAM_USER_LIBBIN/libmetalPCG.so`.

## Usage

In `system/controlDict`:

```cpp
libs ("libmetalPCG.so");
```

In `system/fvSolution`:

```cpp
solvers
{
    p
    {
        solver          metalPCG;
        preconditioner
        {
            preconditioner  GAMG;
            smoother        MetalGCGS;
            agglomerator    faceAreaPair;
            mergeLevels     1;
            nVcycles        2;
        }
        tolerance       1e-06;
        relTol          0.1;
    }
}
```

`metalPCG` accepts all the same controls as `PCG`. `MetalGCGS` can be used anywhere `GaussSeidel` is valid as a GAMG smoother.

---

## Architecture

### metalPCG solver

The pressure solve follows a hybrid CPU/GPU split:

```
GPU: SpMV (A*p)          — Metal gs_color_sweep / sparseMV kernel
GPU: dot products         — partial reduction kernels, summed on CPU
GPU: SAXPY updates        — saxpy / saxpy_sub kernels
CPU: preconditioner       — GAMG V-cycle (restriction, coarse solve, prolongation)
CPU: convergence check    — gSumMag, normFactor
CPU: MPI halo exchange    — initMatrixInterfaces / updateMatrixInterfaces
```

The matrix is uploaded once per mesh change in CSR format (float32). A fingerprint `(nRows, nnz, valSum)` detects matrix changes without re-uploading unnecessarily.

In serial, the full PCG inner loop runs on GPU (3 Metal command buffers per iteration). In parallel (MPI), the GPU handles the local SpMV while MPI halo exchange runs concurrently, then interface contributions are added on CPU before the next iteration.

### MetalGCGS smoother

Replaces GaussSeidel inside GAMG V-cycles at the fine level.

**Graph coloring:** On matrix upload, a greedy graph coloring of the CSR adjacency graph partitions rows into ~6–8 colors (for structured hex meshes). Rows sharing the same color have no edges between them — their GaussSeidel updates are race-free and can execute in parallel.

**Kernel:** One `gs_color_sweep` Metal dispatch per color. Each thread computes:
```
x[i] = (b[i] - sum_{j≠i} A_ij * x[j]) / A_ii
```
A forward pass (color 0 → N) followed by a backward pass (color N → 0) gives symmetric GaussSeidel — the same smoothing quality as serial GS, which GAMG requires for fast convergence.

**Float32 note:** Unlike Jacobi (which accumulates corrections), GaussSeidel replaces `x[i]` outright each sweep. Float32 rounding does not compound across sweeps, so convergence to 1e-6 absolute tolerance is achieved cleanly despite Metal's float32-only GPU.

**Parallel fallback:** When processor interfaces are present (MPI parallel run), the smoother delegates to `GaussSeidelSmoother::smooth()` on CPU. This preserves GAMG convergence quality — same iteration counts as vanilla.

---

## Results

All benchmarks: 1M-cell lid-driven cavity, incompressible laminar, 50 time steps, Apple Silicon.

### Serial (1 MPI rank)

| Solver | Preconditioner | Smoother | Time |
|--------|---------------|----------|------|
| PCG | DIC | — | 253.2s |
| metalPCG v1 (GPU SpMV) | DIC | — | 188.7s (+25%) |
| metalPCG v2 (full GPU loop) | DIC | — | 173.4s (+32%) |
| PCG | GAMG | GaussSeidel | 58.3s |
| metalPCG v1 | GAMG | GaussSeidel | 59.6s |
| metalPCG v2 | GAMG | GaussSeidel | 59.0s |
| **metalPCG v2** | **GAMG** | **MetalGCGS** | **50.5s (+15%)** |

### Parallel (8 MPI ranks, scotch decomposition)

| Solver | Smoother | Time |
|--------|----------|------|
| PCG (CPU only) | GaussSeidel | 14.2s |
| metalPCG (GPU SpMV + CPU GS fallback) | MetalGCGS | ~14–15s |

---

## Key findings

**Preconditioner choice dominates.** Switching from DIC to GAMG reduced runtime from 253s to 58s — a 4× win with no GPU involvement. This dwarfs any GPU benefit and should be the first thing any user optimises.

**GPU SpMV helps most with many iterations.** With DIC (~220 iterations/step), GPU SpMV gives 25–32%. With GAMG (1–9 iterations/step), the SpMV is a small fraction of total time and GPU barely helps.

**Jacobi smoothing quality is the limiting factor for GPU GAMG.** A naive GPU smoother using damped Jacobi ran at 107s — slower than vanilla — because 1 Jacobi sweep ≈ 0.25 GaussSeidel sweeps in error reduction. GAMG needed 3–5× more outer PCG iterations to compensate. Graph-colored GaussSeidel solves this by preserving GS's replacement semantics while being GPU-parallelisable.

**MPI parallelism outperforms single-rank GPU acceleration.** 8-rank CPU ran in 14.2s vs 50.5s for single-rank GPU — 3.5× faster. The CPU performance cores on Apple Silicon are extremely fast, and MPI scaling is straightforward for this problem size.

**GPU doesn't help in the parallel case.** 8 MPI ranks each have a separate OS process and a separate `MTLCommandQueue`. Metal has no multi-process concurrent execution (unlike NVIDIA's CUDA MPS). The 8 queues serialize on the one physical GPU, so command queue contention cancels the per-rank SpMV speedup. metalPCG matches but does not beat vanilla at 8 ranks.

**Unified memory limits the GPU bandwidth advantage.** On Apple Silicon, CPU and GPU share the same DRAM. There is no bandwidth cliff between them as there is between discrete GPU HBM and CPU DDR on a workstation. SpMV is memory-bandwidth-bound; without a bandwidth advantage, the GPU's benefit is smaller than on NVIDIA hardware.

---

## When to use metalPCG

**Worth using:** Single-rank serial runs where MPI is not available or practical. You get 15% over vanilla PCG+GAMG for free, with no change to your case setup beyond the solver name and `libs` entry.

**Not worth using:** Parallel MPI runs — just use vanilla PCG+GAMG with as many ranks as you have cores. The CPU parallelism wins by a wide margin.

**Not applicable:** Cases requiring float64 precision throughout — Metal has no float64, so the GPU path operates in float32. The solver self-corrects (PCG is float64 on CPU; GPU handles only SpMV and inner-loop arithmetic), but if bit-identical results are a hard requirement this is not suitable.

---

## Limitations and future directions

- **Float64:** Metal Apple Silicon has no float64 hardware. All GPU kernels run float32. The PCG outer loop and convergence checks remain float64 on CPU.
- **Serial only for full benefit:** GPU advantage disappears at 8 MPI ranks due to Metal's single-process GPU model.
- **GAMG coarse levels:** Only the fine-level GAMG smoother is GPU-accelerated. Coarse-level restriction, prolongation, and direct solves remain on CPU.
- **Graph coloring cost:** Greedy coloring at matrix upload is O(n) but adds latency on first solve after mesh change.

If this approach were to be taken further, the most impactful directions would be: graph-colored GaussSeidel for parallel MPI (requires halo-aware coloring across rank boundaries), or porting to a platform with a discrete GPU where the bandwidth advantage makes GPU SpMV significantly more compelling.

---

## Directory structure

```
src/metalPCGSolver/
  MetalPCGSolver.H       — lduMatrix::solver subclass, registered as metalPCG
  MetalPCGSolver.C       — solve(): hybrid GPU PCG loop / CPU fallback
  MetalGCGSSmoother.H    — lduMatrix::smoother subclass, registered as MetalGCGS
  MetalGCGSSmoother.C    — smooth(): GPU GCGS (serial) / CPU GS (parallel)
  metalSpMV.h            — Metal backend public interface
  metalSpMV.mm           — Obj-C++: Metal device, kernels, CSR upload, PCG loop, GCGS
  Make/
    files                — wmake source list
    options              — link flags (-framework Metal -framework Foundation)
tutorials/
  cavity-metal/          — 100K cell cavity case (metalPCG, DIC)
  benchmark-1m-gamg/     — 1M cell cavity case (metalPCG, GAMG+MetalGCGS)
tests/
  residual_check.py      — compare residual history vs vanilla PCG
```
