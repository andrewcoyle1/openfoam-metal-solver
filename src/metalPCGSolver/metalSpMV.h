#pragma once

// Plain C++ interface — no Objective-C types exposed.
// Implementation is in metalSpMV.mm (Objective-C++/Metal).

class MetalSpMV
{
public:
    MetalSpMV();
    ~MetalSpMV();

    // Upload the sparse matrix in CSR format.
    // Double values are converted to float32 internally (Apple Silicon
    // Metal shaders have no float64 hardware).
    void setup
    (
        int nRows,
        const int*    rowPtr,   // length nRows+1
        const int*    colIdx,   // length nnz
        const double* values,   // length nnz
        int nnz
    );

    // y = A*x  (float32 on GPU, result widened back to double)
    void multiply(double* y, const double* x, int n);

    bool isReady() const;

    // Returns the process-wide cached SpMV instance held in MetalShared.
    // Callers should call setup() only when the matrix fingerprint has changed.
    static MetalSpMV* sharedCached();

private:
    struct Impl;
    Impl* impl_;
};
