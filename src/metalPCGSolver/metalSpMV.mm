#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metalSpMV.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

// ---------------------------------------------------------------------------
// Metal shader source — all kernels compiled at first use.
// ---------------------------------------------------------------------------
static const char* kShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void sparseMV(
    device const int*   rowPtr [[ buffer(0) ]],
    device const int*   colIdx [[ buffer(1) ]],
    device const float* vals   [[ buffer(2) ]],
    device const float* x      [[ buffer(3) ]],
    device       float* y      [[ buffer(4) ]],
    uint row [[ thread_position_in_grid ]]
) {
    float sum = 0.0f;
    for (int i = rowPtr[row]; i < rowPtr[row+1]; ++i)
        sum += vals[i] * x[colIdx[i]];
    y[row] = sum;
}

// Parallel dot product — each threadgroup reduces its tile into partials[gidx].
// Caller sums the partials[] array on CPU (nGroups elements, negligible cost).
kernel void dot_partial(
    device const float* a        [[ buffer(0) ]],
    device const float* b        [[ buffer(1) ]],
    device       float* partials [[ buffer(2) ]],
    constant     int&   n        [[ buffer(3) ]],
    uint lid  [[ thread_position_in_threadgroup ]],
    uint gid  [[ thread_position_in_grid ]],
    uint tgs  [[ threads_per_threadgroup ]],
    uint gidx [[ threadgroup_position_in_grid ]],
    threadgroup float* smem [[ threadgroup(0) ]]
) {
    smem[lid] = (gid < (uint)n) ? a[gid] * b[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[gidx] = smem[0];
}

// sum(|a[i]|) — same two-pass reduction, used for residual norm.
kernel void sumabs_partial(
    device const float* a        [[ buffer(0) ]],
    device       float* partials [[ buffer(1) ]],
    constant     int&   n        [[ buffer(2) ]],
    uint lid  [[ thread_position_in_threadgroup ]],
    uint gid  [[ thread_position_in_grid ]],
    uint tgs  [[ threads_per_threadgroup ]],
    uint gidx [[ threadgroup_position_in_grid ]],
    threadgroup float* smem [[ threadgroup(0) ]]
) {
    smem[lid] = (gid < (uint)n) ? abs(a[gid]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[gidx] = smem[0];
}

// y += alpha * x
kernel void saxpy(
    device       float* y      [[ buffer(0) ]],
    device const float* x      [[ buffer(1) ]],
    constant     float& alpha  [[ buffer(2) ]],
    uint idx [[ thread_position_in_grid ]]
) {
    y[idx] += alpha * x[idx];
}

// y -= alpha * x
kernel void saxpy_sub(
    device       float* y      [[ buffer(0) ]],
    device const float* x      [[ buffer(1) ]],
    constant     float& alpha  [[ buffer(2) ]],
    uint idx [[ thread_position_in_grid ]]
) {
    y[idx] -= alpha * x[idx];
}

// p = z + beta * p  (PCG search-direction update)
kernel void pcg_p_update(
    device       float* p     [[ buffer(0) ]],
    device const float* z     [[ buffer(1) ]],
    constant     float& beta  [[ buffer(2) ]],
    uint idx [[ thread_position_in_grid ]]
) {
    p[idx] = z[idx] + beta * p[idx];
}

// Graph-colored Gauss-Seidel sweep: one thread per row of the current color.
// colorRows[gid] gives the actual row index; threads within a color share no
// edges so writes to x[i] are race-free.
kernel void gs_color_sweep(
    device const int*   rowPtr    [[ buffer(0) ]],
    device const int*   colIdx    [[ buffer(1) ]],
    device const float* vals      [[ buffer(2) ]],
    device const float* source    [[ buffer(3) ]],
    device       float* x         [[ buffer(4) ]],
    device const int*   colorRows [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    const int i = colorRows[gid];
    float accum = source[i];
    float diag  = 1.0f;
    const int end = rowPtr[i + 1];
    for (int k = rowPtr[i]; k < end; ++k)
    {
        const int j = colIdx[k];
        if (j == i) diag = vals[k];
        else        accum -= vals[k] * x[j];
    }
    x[i] = accum / diag;
}
)";

// ---------------------------------------------------------------------------
// Process-wide Metal state — initialised once.
// ---------------------------------------------------------------------------
struct MetalShared
{
    id<MTLDevice>               device       = nil;
    id<MTLCommandQueue>         queue        = nil;
    id<MTLComputePipelineState> pipeSpMV     = nil;
    id<MTLComputePipelineState> pipeDot      = nil;
    id<MTLComputePipelineState> pipeSumAbs   = nil;
    id<MTLComputePipelineState> pipeSaxpy    = nil;
    id<MTLComputePipelineState> pipeSaxpySub = nil;
    id<MTLComputePipelineState> pipePUpdate  = nil;
    id<MTLComputePipelineState> pipeGSColor  = nil;
    bool ok = false;

    MetalSpMV cachedSpMV;

    static MetalShared& instance()
    {
        static MetalShared s;
        return s;
    }

private:
    MetalShared()
    {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        queue = [device newCommandQueue];

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kShaderSrc];
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:[MTLCompileOptions new]
                                                    error:&err];
        if (!lib) { NSLog(@"MetalSpMV shader: %@", err.localizedDescription); return; }

        auto makePipe = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            NSError* e = nil;
            id<MTLComputePipelineState> p =
                [device newComputePipelineStateWithFunction:fn error:&e];
            if (!p) NSLog(@"MetalSpMV pipeline %@: %@", name, e.localizedDescription);
            return p;
        };

        pipeSpMV     = makePipe(@"sparseMV");
        pipeDot      = makePipe(@"dot_partial");
        pipeSumAbs   = makePipe(@"sumabs_partial");
        pipeSaxpy    = makePipe(@"saxpy");
        pipeSaxpySub = makePipe(@"saxpy_sub");
        pipePUpdate  = makePipe(@"pcg_p_update");
        pipeGSColor  = makePipe(@"gs_color_sweep");

        ok = pipeSpMV && pipeDot && pipeSumAbs &&
             pipeSaxpy && pipeSaxpySub && pipePUpdate && pipeGSColor;
    }
};

// ---------------------------------------------------------------------------
// Per-instance implementation data.
// ---------------------------------------------------------------------------
struct MetalSpMV::Impl
{
    // Sparse matrix (CSR, float32)
    id<MTLBuffer> rowPtrBuf  = nil;
    id<MTLBuffer> colIdxBuf  = nil;
    id<MTLBuffer> valuesBuf  = nil;
    int    nRows    = -1;
    int    nnz      = -1;
    double valSum   = 0.0;
    bool   ready    = false;

    // Vector buffers for multiply() — reused across calls
    id<MTLBuffer> xBuf    = nil;
    id<MTLBuffer> yBuf    = nil;
    int    cachedN  = -1;

    // PCG GPU solve buffers — allocated once per mesh size
    id<MTLBuffer> rBuf       = nil;  // residual (float32)
    id<MTLBuffer> pBuf       = nil;  // search direction (float32)
    id<MTLBuffer> zBuf       = nil;  // preconditioned residual (float32)
    id<MTLBuffer> ApBuf      = nil;  // A*p (float32)
    id<MTLBuffer> partialsBuf = nil; // partial sums (float32, nGroups)
    id<MTLBuffer> nBuf       = nil;  // constant int n
    id<MTLBuffer> scalarBuf  = nil;  // float scalar (alpha or beta)
    int    cachedPCGN = -1;
    int    nGroups    = 0;

    // Graph-colored Gauss-Seidel smoother buffers
    int                      nColors     = 0;
    std::vector<id<MTLBuffer>> colorRowBufs;  // one per color: int[] row indices
    std::vector<int>           colorSizes;    // number of rows per color
    id<MTLBuffer>            xGCGSBuf    = nil;  // float32 x (solution)
    id<MTLBuffer>            srcGCGSBuf  = nil;  // float32 source (rhs)
    bool                     gcgsReady   = false;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float sumPartials(const id<MTLBuffer> buf, int nGroups)
{
    const float* p = static_cast<const float*>([buf contents]);
    float s = 0.0f;
    for (int i = 0; i < nGroups; ++i) s += p[i];
    return s;
}

static void encodeSpMV(
    id<MTLComputeCommandEncoder> enc,
    MetalShared& m,
    id<MTLBuffer> rowPtrBuf,
    id<MTLBuffer> colIdxBuf,
    id<MTLBuffer> valuesBuf,
    id<MTLBuffer> xBuf,
    id<MTLBuffer> yBuf,
    int n)
{
    NSUInteger tg = std::min((NSUInteger)256,
                             m.pipeSpMV.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:m.pipeSpMV];
    [enc setBuffer:rowPtrBuf offset:0 atIndex:0];
    [enc setBuffer:colIdxBuf offset:0 atIndex:1];
    [enc setBuffer:valuesBuf offset:0 atIndex:2];
    [enc setBuffer:xBuf      offset:0 atIndex:3];
    [enc setBuffer:yBuf      offset:0 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void encodeDot(
    id<MTLComputeCommandEncoder> enc,
    MetalShared& m,
    id<MTLBuffer> a,
    id<MTLBuffer> b,
    id<MTLBuffer> partials,
    id<MTLBuffer> nBuf,
    int n,
    int nGroups)
{
    const NSUInteger tg = 256;
    [enc setComputePipelineState:m.pipeDot];
    [enc setBuffer:a        offset:0 atIndex:0];
    [enc setBuffer:b        offset:0 atIndex:1];
    [enc setBuffer:partials offset:0 atIndex:2];
    [enc setBuffer:nBuf     offset:0 atIndex:3];
    [enc setThreadgroupMemoryLength:tg * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nGroups,1,1)
        threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void encodeSumAbs(
    id<MTLComputeCommandEncoder> enc,
    MetalShared& m,
    id<MTLBuffer> a,
    id<MTLBuffer> partials,
    id<MTLBuffer> nBuf,
    int n,
    int nGroups)
{
    const NSUInteger tg = 256;
    [enc setComputePipelineState:m.pipeSumAbs];
    [enc setBuffer:a        offset:0 atIndex:0];
    [enc setBuffer:partials offset:0 atIndex:1];
    [enc setBuffer:nBuf     offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:tg * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(nGroups,1,1)
        threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

static void encodeSaxpy(
    id<MTLComputeCommandEncoder> enc,
    MetalShared& m,
    id<MTLBuffer> y,
    id<MTLBuffer> x,
    id<MTLBuffer> alpha,
    int n,
    bool sub = false)
{
    NSUInteger tg = std::min((NSUInteger)256,
        (sub ? m.pipeSaxpySub : m.pipeSaxpy).maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:sub ? m.pipeSaxpySub : m.pipeSaxpy];
    [enc setBuffer:y     offset:0 atIndex:0];
    [enc setBuffer:x     offset:0 atIndex:1];
    [enc setBuffer:alpha offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
}

// ---------------------------------------------------------------------------
// MetalSpMV implementation
// ---------------------------------------------------------------------------

MetalSpMV::MetalSpMV() : impl_(new Impl()) {}

MetalSpMV::~MetalSpMV()
{
    @autoreleasepool
    {
        impl_->rowPtrBuf  = nil;
        impl_->colIdxBuf  = nil;
        impl_->valuesBuf  = nil;
        impl_->xBuf       = nil;
        impl_->yBuf       = nil;
        impl_->rBuf       = nil;
        impl_->pBuf       = nil;
        impl_->zBuf       = nil;
        impl_->ApBuf      = nil;
        impl_->partialsBuf = nil;
        impl_->nBuf       = nil;
        impl_->scalarBuf  = nil;
        impl_->xGCGSBuf   = nil;
        impl_->srcGCGSBuf = nil;
        for (auto& b : impl_->colorRowBufs) b = nil;
        impl_->colorRowBufs.clear();
    }
    delete impl_;
}

void MetalSpMV::setup(int nRows, const int* rowPtr, const int* colIdx,
                      const double* values, int nnz)
{
    MetalShared& m = MetalShared::instance();
    if (!m.ok) return;

    double valSum = 0.0;
    const int fpN = std::min(8, nnz);
    for (int i = 0; i < fpN; ++i) valSum += values[i];

    if (impl_->nRows == nRows && impl_->nnz == nnz && impl_->valSum == valSum)
        return;

    impl_->ready = false;

    std::vector<float> fvals(nnz);
    for (int i = 0; i < nnz; ++i) fvals[i] = float(values[i]);

    @autoreleasepool
    {
        impl_->rowPtrBuf = [m.device newBufferWithBytes:rowPtr
                                                 length:(nRows+1)*sizeof(int)
                                                options:MTLResourceStorageModeShared];
        impl_->colIdxBuf = [m.device newBufferWithBytes:colIdx
                                                 length:nnz*sizeof(int)
                                                options:MTLResourceStorageModeShared];
        impl_->valuesBuf = [m.device newBufferWithBytes:fvals.data()
                                                 length:nnz*sizeof(float)
                                                options:MTLResourceStorageModeShared];
    }

    impl_->ready  = (impl_->rowPtrBuf && impl_->colIdxBuf && impl_->valuesBuf);
    impl_->nRows  = nRows;
    impl_->nnz    = nnz;
    impl_->valSum = valSum;

    if (impl_->ready)
        setupGCGS(nRows, rowPtr, colIdx, values, nnz);
}

// Greedy graph coloring + GPU buffer setup for graph-colored GaussSeidel.
// Each color is a maximal independent set in the adjacency graph → rows of the
// same color share no edges → their GS updates are race-free in parallel.
void MetalSpMV::setupGCGS(int nRows, const int* rowPtr, const int* colIdx,
                           const double* /*values*/, int /*nnz*/)
{
    MetalShared& m = MetalShared::instance();
    impl_->gcgsReady   = false;
    impl_->colorRowBufs.clear();
    impl_->colorSizes.clear();
    impl_->nColors     = 0;

    // Greedy coloring: for each row, assign the smallest color not used by any
    // neighbour that has already been colored.
    std::vector<int> color(nRows, -1);
    std::vector<bool> forbidden;
    int nColors = 0;
    for (int i = 0; i < nRows; ++i)
    {
        forbidden.assign(nColors, false);
        for (int k = rowPtr[i]; k < rowPtr[i+1]; ++k)
        {
            int j = colIdx[k];
            if (j != i && color[j] >= 0)
            {
                if (color[j] >= (int)forbidden.size())
                    forbidden.resize(color[j]+1, false);
                forbidden[color[j]] = true;
            }
        }
        int c = 0;
        while (c < (int)forbidden.size() && forbidden[c]) ++c;
        color[i] = c;
        if (c >= nColors) nColors = c + 1;
    }

    // Bucket rows by color.
    std::vector<std::vector<int>> buckets(nColors);
    for (int i = 0; i < nRows; ++i) buckets[color[i]].push_back(i);

    // Upload one MTLBuffer per color containing its row indices.
    impl_->colorRowBufs.resize(nColors, nil);
    impl_->colorSizes.resize(nColors, 0);
    for (int c = 0; c < nColors; ++c)
    {
        const auto& rows = buckets[c];
        impl_->colorSizes[c] = (int)rows.size();
        impl_->colorRowBufs[c] = [m.device
            newBufferWithBytes:rows.data()
                        length:rows.size()*sizeof(int)
                       options:MTLResourceStorageModeShared];
    }

    // Working x and source buffers (float32, nRows each).
    impl_->xGCGSBuf   = [m.device newBufferWithLength:nRows*sizeof(float)
                                              options:MTLResourceStorageModeShared];
    impl_->srcGCGSBuf = [m.device newBufferWithLength:nRows*sizeof(float)
                                              options:MTLResourceStorageModeShared];

    impl_->nColors   = nColors;
    impl_->gcgsReady = impl_->xGCGSBuf && impl_->srcGCGSBuf;
}

void MetalSpMV::multiply(double* y, const double* x, int n)
{
    if (!impl_->ready) return;
    MetalShared& m = MetalShared::instance();

    if (impl_->cachedN != n)
    {
        impl_->xBuf = [m.device newBufferWithLength:n*sizeof(float)
                                            options:MTLResourceStorageModeShared];
        impl_->yBuf = [m.device newBufferWithLength:n*sizeof(float)
                                            options:MTLResourceStorageModeShared];
        impl_->cachedN = n;
    }

    @autoreleasepool
    {
        float* fx = static_cast<float*>([impl_->xBuf contents]);
        for (int i = 0; i < n; ++i) fx[i] = float(x[i]);

        id<MTLCommandBuffer>         cmd = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encodeSpMV(enc, m,
                   impl_->rowPtrBuf, impl_->colIdxBuf, impl_->valuesBuf,
                   impl_->xBuf, impl_->yBuf, n);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const float* fy = static_cast<const float*>([impl_->yBuf contents]);
        for (int i = 0; i < n; ++i) y[i] = double(fy[i]);
    }
}

bool MetalSpMV::isReady()  const { return impl_->ready; }
int  MetalSpMV::nRows()    const { return impl_->nRows; }
bool MetalSpMV::hasGCGS()  const { return impl_->gcgsReady; }

// Graph-colored Gauss-Seidel: one forward + one backward pass per sweep.
// Each color launches one Metal kernel dispatch; rows within a color are
// edge-independent so writes to x are race-free.
void MetalSpMV::smoothGCGS(double* x, const double* source, int n, int nSweeps)
{
    if (!impl_->gcgsReady || n != impl_->nRows) return;
    MetalShared& m = MetalShared::instance();
    NSUInteger tg = std::min((NSUInteger)256,
                             m.pipeGSColor.maxTotalThreadsPerThreadgroup);

    // Copy double→float into GPU-shared buffers.
    float* xf  = static_cast<float*>([impl_->xGCGSBuf   contents]);
    float* sf  = static_cast<float*>([impl_->srcGCGSBuf  contents]);
    for (int i = 0; i < n; ++i) { xf[i] = float(x[i]); sf[i] = float(source[i]); }

    for (int sweep = 0; sweep < nSweeps; ++sweep)
    {
        // Forward pass: color 0 → nColors-1
        for (int c = 0; c < impl_->nColors; ++c)
        {
            int sz = impl_->colorSizes[c];
            id<MTLCommandBuffer> cb = [m.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:m.pipeGSColor];
            [enc setBuffer:impl_->rowPtrBuf           offset:0 atIndex:0];
            [enc setBuffer:impl_->colIdxBuf           offset:0 atIndex:1];
            [enc setBuffer:impl_->valuesBuf           offset:0 atIndex:2];
            [enc setBuffer:impl_->srcGCGSBuf          offset:0 atIndex:3];
            [enc setBuffer:impl_->xGCGSBuf            offset:0 atIndex:4];
            [enc setBuffer:impl_->colorRowBufs[c]     offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(sz,1,1)
                threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        // Backward pass: color nColors-1 → 0 (symmetric GS)
        for (int c = impl_->nColors - 1; c >= 0; --c)
        {
            int sz = impl_->colorSizes[c];
            id<MTLCommandBuffer> cb = [m.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:m.pipeGSColor];
            [enc setBuffer:impl_->rowPtrBuf           offset:0 atIndex:0];
            [enc setBuffer:impl_->colIdxBuf           offset:0 atIndex:1];
            [enc setBuffer:impl_->valuesBuf           offset:0 atIndex:2];
            [enc setBuffer:impl_->srcGCGSBuf          offset:0 atIndex:3];
            [enc setBuffer:impl_->xGCGSBuf            offset:0 atIndex:4];
            [enc setBuffer:impl_->colorRowBufs[c]     offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(sz,1,1)
                threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    // Copy float→double back to CPU.
    for (int i = 0; i < n; ++i) x[i] = double(xf[i]);
}

MetalSpMV* MetalSpMV::sharedCached()
{
    return &MetalShared::instance().cachedSpMV;
}

// ---------------------------------------------------------------------------
// GPU PCG solve
// ---------------------------------------------------------------------------

bool MetalSpMV::solveGPUPCG
(
    double*       psi,
    const double* b,
    int           n,
    double        normFactor,
    double        tolerance,
    double        relTol,
    int           maxIter,
    int           minIter,
    std::function<void(double* w, const double* r, int n)> precond,
    double& initialResidual,
    double& finalResidual,
    int&    nIterations
)
{
    if (!impl_->ready) return false;
    MetalShared& m = MetalShared::instance();
    if (!m.ok) return false;

    // Allocate or reuse PCG vector buffers.
    if (impl_->cachedPCGN != n)
    {
        const NSUInteger sz = n * sizeof(float);
        const int ng = (n + 255) / 256;
        impl_->rBuf        = [m.device newBufferWithLength:sz
                                                   options:MTLResourceStorageModeShared];
        impl_->pBuf        = [m.device newBufferWithLength:sz
                                                   options:MTLResourceStorageModeShared];
        impl_->zBuf        = [m.device newBufferWithLength:sz
                                                   options:MTLResourceStorageModeShared];
        impl_->ApBuf       = [m.device newBufferWithLength:sz
                                                   options:MTLResourceStorageModeShared];
        impl_->partialsBuf = [m.device newBufferWithLength:ng*sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        impl_->nBuf        = [m.device newBufferWithBytes:&n
                                                   length:sizeof(int)
                                                  options:MTLResourceStorageModeShared];
        impl_->scalarBuf   = [m.device newBufferWithLength:sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        impl_->cachedPCGN  = n;
        impl_->nGroups     = ng;
    }

    const int ng = impl_->nGroups;

    // Upload initial psi to xBuf (reuse multiply()'s buffer, allocate if needed).
    if (impl_->cachedN != n)
    {
        impl_->xBuf = [m.device newBufferWithLength:n*sizeof(float)
                                            options:MTLResourceStorageModeShared];
        impl_->yBuf = [m.device newBufferWithLength:n*sizeof(float)
                                            options:MTLResourceStorageModeShared];
        impl_->cachedN = n;
    }

    // Write psi (double→float) into xBuf; compute r0 = b - A*psi on CPU,
    // then upload r0 (double→float) into rBuf.  The CPU r0 is already available
    // from the caller (we receive initialResidual from outside), so we only need
    // to upload the caller-supplied psi and compute A*psi on GPU for internal use.
    // Simpler: upload b and psi, run GPU SpMV for Ap, subtract on CPU, upload r.
    // Even simpler: caller already has r0 as a double array in 'b - A*psi'.
    // We receive it implicitly because initialResidual is set before this call.
    // We recompute r0 on GPU to have it in rBuf (float32).

    @autoreleasepool
    {
        // Write psi into xBuf
        float* fpsi = static_cast<float*>([impl_->xBuf contents]);
        for (int i = 0; i < n; ++i) fpsi[i] = float(psi[i]);

        // Write b into yBuf temporarily (reusing as b-buffer)
        float* fb = static_cast<float*>([impl_->yBuf contents]);
        for (int i = 0; i < n; ++i) fb[i] = float(b[i]);

        // CB: Ap = A*psi (xBuf→yBuf will be overwritten), store in ApBuf
        //     r = b - Ap (done element-wise on GPU using rBuf, yBuf=b, ApBuf)
        // Encode: SpMV(xBuf → ApBuf), then r = b - Ap on CPU (shared mem, fast)
        id<MTLCommandBuffer>         cmd = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encodeSpMV(enc, m,
                   impl_->rowPtrBuf, impl_->colIdxBuf, impl_->valuesBuf,
                   impl_->xBuf, impl_->ApBuf, n);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // CPU: r = b - Ap using shared memory pointers (no real transfer on AS)
        const float* fAp = static_cast<const float*>([impl_->ApBuf contents]);
        float* fr  = static_cast<float*>([impl_->rBuf contents]);
        double r0norm = 0.0;
        for (int i = 0; i < n; ++i)
        {
            fr[i] = fb[i] - fAp[i];
            r0norm += std::fabs(double(fr[i]));
        }
        initialResidual = r0norm / normFactor;
        finalResidual   = initialResidual;
        nIterations     = 0;

        if (initialResidual < tolerance)
            return true;  // already converged, nothing to do

        // PCG loop
        // Per-iteration structure:
        //   CPU: precond(z = M^{-1}r) — reads rBuf, writes zBuf (shared memory)
        //   CB_dot: wArA = dot(z, r)
        //   wait; CPU: compute beta; write scalarBuf
        //   CB_main: p = z+beta*p (or p=z if first); Ap=A*p; wApA=dot(Ap,p)
        //   wait; CPU: compute alpha; write scalarBuf
        //   CB_update: psi+=alpha*p; r-=alpha*Ap; norm=sumabs(r)
        //   wait; CPU: convergence check

        double wArAold = 1.0;  // dummy — only beta=wArA/wArAold matters and
                               // on first iteration we set p=z (beta unused)

        std::vector<double> rDbl(n), wDbl(n);

        for (int iter = 0; iter < maxIter; ++iter)
        {
            // Read r from GPU shared memory → double, run preconditioner → w double,
            // write w back to GPU shared memory as float.
            const float* fr2 = static_cast<const float*>([impl_->rBuf contents]);
            for (int i = 0; i < n; ++i) rDbl[i] = double(fr2[i]);

            precond(wDbl.data(), rDbl.data(), n);

            float* fz = static_cast<float*>([impl_->zBuf contents]);
            for (int i = 0; i < n; ++i) fz[i] = float(wDbl[i]);

            // CB_dot: wArA = dot(zBuf, rBuf)
            {
                id<MTLCommandBuffer>         cmd2 = [m.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
                encodeDot(enc2, m, impl_->zBuf, impl_->rBuf,
                          impl_->partialsBuf, impl_->nBuf, n, ng);
                [enc2 endEncoding];
                [cmd2 commit];
                [cmd2 waitUntilCompleted];
            }
            double wArA = double(sumPartials(impl_->partialsBuf, ng));

            float beta = (iter == 0) ? 0.0f : float(wArA / wArAold);
            float* fscalar = static_cast<float*>([impl_->scalarBuf contents]);
            *fscalar = beta;

            // CB_main: p = z + beta*p (or p=z);  Ap=A*p;  wApA=dot(Ap,p)
            {
                id<MTLCommandBuffer>         cmd3 = [m.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc3 = [cmd3 computeCommandEncoder];

                // p = z + beta*p
                NSUInteger tg = std::min((NSUInteger)256,
                    m.pipePUpdate.maxTotalThreadsPerThreadgroup);
                [enc3 setComputePipelineState:m.pipePUpdate];
                [enc3 setBuffer:impl_->pBuf    offset:0 atIndex:0];
                [enc3 setBuffer:impl_->zBuf    offset:0 atIndex:1];
                [enc3 setBuffer:impl_->scalarBuf offset:0 atIndex:2];
                [enc3 dispatchThreads:MTLSizeMake(n,1,1)
                    threadsPerThreadgroup:MTLSizeMake(tg,1,1)];

                // Ap = A*p
                encodeSpMV(enc3, m,
                           impl_->rowPtrBuf, impl_->colIdxBuf, impl_->valuesBuf,
                           impl_->pBuf, impl_->ApBuf, n);

                // wApA = dot(Ap, p)
                encodeDot(enc3, m, impl_->ApBuf, impl_->pBuf,
                          impl_->partialsBuf, impl_->nBuf, n, ng);

                [enc3 endEncoding];
                [cmd3 commit];
                [cmd3 waitUntilCompleted];
            }
            double wApA = double(sumPartials(impl_->partialsBuf, ng));

            if (std::fabs(wApA) < 1e-30) break;  // singular

            float alpha = float(wArA / wApA);
            *fscalar = alpha;

            // CB_update: psi+=alpha*p;  r-=alpha*Ap;  norm=sumabs(r)
            {
                id<MTLCommandBuffer>         cmd4 = [m.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc4 = [cmd4 computeCommandEncoder];

                // Reuse xBuf as psi buffer for GPU solve (uploaded at start)
                encodeSaxpy(enc4, m, impl_->xBuf, impl_->pBuf, impl_->scalarBuf, n);
                encodeSaxpy(enc4, m, impl_->rBuf,  impl_->ApBuf, impl_->scalarBuf, n, true);
                encodeSumAbs(enc4, m, impl_->rBuf, impl_->partialsBuf,
                             impl_->nBuf, n, ng);

                [enc4 endEncoding];
                [cmd4 commit];
                [cmd4 waitUntilCompleted];
            }

            double norm = double(sumPartials(impl_->partialsBuf, ng)) / normFactor;
            finalResidual = norm;
            nIterations   = iter + 1;
            wArAold = wArA;

            bool converged = (norm < tolerance) ||
                             (norm / (initialResidual + 1e-30) < relTol);
            if (converged && (iter + 1) >= minIter)
                break;
        }

        // Download psi from xBuf (shared memory → double)
        const float* fpsi2 = static_cast<const float*>([impl_->xBuf contents]);
        for (int i = 0; i < n; ++i) psi[i] = double(fpsi2[i]);
    }

    return true;
}
