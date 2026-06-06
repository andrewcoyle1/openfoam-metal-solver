#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metalSpMV.h"
#include <vector>
#include <algorithm>

static const char* kSpMVShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void sparseMV(
    device const int*   rowPtr [[ buffer(0) ]],
    device const int*   colIdx [[ buffer(1) ]],
    device const float* vals   [[ buffer(2) ]],
    device const float* x      [[ buffer(3) ]],
    device       float* y      [[ buffer(4) ]],
    uint row [[ thread_position_in_grid ]]
)
{
    float sum = 0.0f;
    const int start = rowPtr[row];
    const int end   = rowPtr[row + 1];
    for (int i = start; i < end; ++i)
        sum += vals[i] * x[colIdx[i]];
    y[row] = sum;
}
)";

// Shared Metal state — initialised once per process.
// Creating a new MTLDevice per solver instance is expensive and leaks
// Metal command buffer contexts; use a process-wide singleton instead.
struct MetalShared
{
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLComputePipelineState> pipeline = nil;
    bool ok = false;

    // Cached CSR matrix — survives across solver instantiations so the GPU
    // buffers are only reallocated when the matrix actually changes.
    // The fingerprint is stored inside cachedSpMV.impl_.
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
        NSString* src = [NSString stringWithUTF8String:kSpMVShaderSrc];
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:[MTLCompileOptions new]
                                                    error:&err];
        if (!lib) { NSLog(@"MetalSpMV shader: %@", err.localizedDescription); return; }

        id<MTLFunction> fn = [lib newFunctionWithName:@"sparseMV"];
        pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipeline) { NSLog(@"MetalSpMV pipeline: %@", err.localizedDescription); return; }

        ok = true;
    }
};

// Per-instance data: matrix buffers plus a fingerprint for cache validation.
// xBuf and yBuf are cached across multiply() calls to avoid per-call allocation.
struct MetalSpMV::Impl
{
    id<MTLBuffer> rowPtrBuf = nil;
    id<MTLBuffer> colIdxBuf = nil;
    id<MTLBuffer> valuesBuf = nil;
    id<MTLBuffer> xBuf      = nil; // input vector, reused each call
    id<MTLBuffer> yBuf      = nil; // output vector, reused each call
    int    nRows    = -1;
    int    nnz      = -1;
    int    cachedN  = -1;          // size of xBuf/yBuf
    double valSum   = 0.0; // sum of first 8 values — lightweight fingerprint
    bool   ready    = false;
};

MetalSpMV::MetalSpMV() : impl_(new Impl()) {}

MetalSpMV::~MetalSpMV()
{
    @autoreleasepool
    {
        impl_->rowPtrBuf = nil;
        impl_->colIdxBuf = nil;
        impl_->valuesBuf = nil;
        impl_->xBuf      = nil;
        impl_->yBuf      = nil;
    }
    delete impl_;
}

void MetalSpMV::setup
(
    int nRows,
    const int*    rowPtr,
    const int*    colIdx,
    const double* values,
    int nnz
)
{
    MetalShared& m = MetalShared::instance();
    if (!m.ok) return;

    // Lightweight fingerprint: sum of first 8 CSR values.  For a fixed mesh
    // with constant coefficients (laminar pressure Poisson), this detects an
    // unchanged matrix in O(1) and avoids reallocating GPU buffers each step.
    double valSum = 0.0;
    const int fpN = std::min(8, nnz);
    for (int i = 0; i < fpN; ++i) valSum += values[i];

    if (impl_->nRows == nRows && impl_->nnz == nnz && impl_->valSum == valSum)
        return; // matrix unchanged — existing GPU buffers remain valid

    impl_->ready = false;

    std::vector<float> fvals(nnz);
    for (int i = 0; i < nnz; ++i)
        fvals[i] = float(values[i]);

    @autoreleasepool
    {
        impl_->rowPtrBuf = [m.device
            newBufferWithBytes:rowPtr
                       length:(nRows + 1) * sizeof(int)
                      options:MTLResourceStorageModeShared];

        impl_->colIdxBuf = [m.device
            newBufferWithBytes:colIdx
                       length:nnz * sizeof(int)
                      options:MTLResourceStorageModeShared];

        impl_->valuesBuf = [m.device
            newBufferWithBytes:fvals.data()
                       length:nnz * sizeof(float)
                      options:MTLResourceStorageModeShared];
    }

    impl_->ready  = (impl_->rowPtrBuf && impl_->colIdxBuf && impl_->valuesBuf);
    impl_->nRows  = nRows;
    impl_->nnz    = nnz;
    impl_->valSum = valSum;
}

void MetalSpMV::multiply(double* y, const double* x, int n)
{
    if (!impl_->ready) return;

    MetalShared& m = MetalShared::instance();

    // Allocate persistent vector buffers on first call or if n changed.
    // MTLResourceStorageModeShared gives direct CPU access — no copy needed.
    if (impl_->cachedN != n)
    {
        impl_->xBuf = [m.device newBufferWithLength:n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        impl_->yBuf = [m.device newBufferWithLength:n * sizeof(float)
                                            options:MTLResourceStorageModeShared];
        impl_->cachedN = n;
    }

    @autoreleasepool
    {
        // Write x into the shared-memory buffer directly (no intermediate vector).
        float* fxPtr = static_cast<float*>([impl_->xBuf contents]);
        for (int i = 0; i < n; ++i) fxPtr[i] = float(x[i]);

        id<MTLBuffer> xBuf = impl_->xBuf;
        id<MTLBuffer> yBuf = impl_->yBuf;

        id<MTLCommandBuffer>         cmd = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:m.pipeline];
        [enc setBuffer:impl_->rowPtrBuf offset:0 atIndex:0];
        [enc setBuffer:impl_->colIdxBuf offset:0 atIndex:1];
        [enc setBuffer:impl_->valuesBuf offset:0 atIndex:2];
        [enc setBuffer:xBuf             offset:0 atIndex:3];
        [enc setBuffer:yBuf             offset:0 atIndex:4];

        NSUInteger tgSize = m.pipeline.maxTotalThreadsPerThreadgroup;
        if (tgSize > 256) tgSize = 256;

        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        const float* fy = static_cast<const float*>([yBuf contents]);
        for (int i = 0; i < n; ++i) y[i] = double(fy[i]);
    }
}

bool MetalSpMV::isReady() const
{
    return impl_->ready;
}

MetalSpMV* MetalSpMV::sharedCached()
{
    return &MetalShared::instance().cachedSpMV;
}
