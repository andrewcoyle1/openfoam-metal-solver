#pragma once
#include <functional>

class MetalSpMV
{
public:
    MetalSpMV();
    ~MetalSpMV();

    // Upload the sparse matrix in CSR format (float32 on GPU).
    void setup
    (
        int nRows,
        const int*    rowPtr,
        const int*    colIdx,
        const double* values,
        int nnz
    );

    // y = A*x  (float32 on GPU, result widened back to double)
    void multiply(double* y, const double* x, int n);

    bool isReady()  const;
    int  nRows()    const;

    // Graph-colored Gauss-Seidel smoother.
    bool hasGCGS()  const;
    void smoothGCGS(double* x, const double* source, int n, int nSweeps);

    // Full GPU PCG solve.  Caller provides the preconditioner as a lambda
    // (runs on CPU; both r and w are double arrays of length n).
    // Returns false if GPU is not ready (caller should fall back to CPU).
    bool solveGPUPCG
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
    );

    // Process-wide cached SpMV instance held in MetalShared.
    static MetalSpMV* sharedCached();

private:
    struct Impl;
    Impl* impl_;

    void setupGCGS(int nRows, const int* rowPtr, const int* colIdx,
                   const double* values, int nnz);
};
