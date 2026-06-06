#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metalSpMV.h"
#include <vector>

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

// Per-instance data: only the uploaded matrix buffers change each time step
struct MetalSpMV::Impl
{
    id<MTLBuffer> rowPtrBuf = nil;
    id<MTLBuffer> colIdxBuf = nil;
    id<MTLBuffer> valuesBuf = nil;
    int  nRows = 0;
    int  nnz   = 0;
    bool ready = false;
};

MetalSpMV::MetalSpMV() : impl_(new Impl()) {}

MetalSpMV::~MetalSpMV()
{
    @autoreleasepool
    {
        impl_->rowPtrBuf = nil;
        impl_->colIdxBuf = nil;
        impl_->valuesBuf = nil;
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
    impl_->ready = false;
    MetalShared& m = MetalShared::instance();
    if (!m.ok) return;

    impl_->nRows = nRows;
    impl_->nnz   = nnz;

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

    impl_->ready = (impl_->rowPtrBuf && impl_->colIdxBuf && impl_->valuesBuf);
}

void MetalSpMV::multiply(double* y, const double* x, int n)
{
    if (!impl_->ready) return;

    MetalShared& m = MetalShared::instance();

    @autoreleasepool
    {
        std::vector<float> fx(n);
        for (int i = 0; i < n; ++i) fx[i] = float(x[i]);

        id<MTLBuffer> xBuf = [m.device
            newBufferWithBytes:fx.data()
                       length:n * sizeof(float)
                      options:MTLResourceStorageModeShared];

        id<MTLBuffer> yBuf = [m.device
            newBufferWithLength:n * sizeof(float)
                       options:MTLResourceStorageModeShared];

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
