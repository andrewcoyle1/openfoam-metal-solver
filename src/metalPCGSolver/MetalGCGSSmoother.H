#ifndef MetalGCGSSmoother_H
#define MetalGCGSSmoother_H

#include "lduMatrix.H"
#include "GaussSeidelSmoother.H"
#include "metalSpMV.h"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class MetalGCGSSmoother Declaration
\*---------------------------------------------------------------------------*/

// Graph-colored Gauss-Seidel smoother for GAMG on Apple Silicon.
//
// At the fine level (matrix size matches the uploaded Metal singleton) each
// GS sweep is dispatched as a sequence of Metal kernel launches — one per
// color.  Rows sharing the same color have no edges between them, so their
// GS updates x[i] = (b[i] - sum_{j≠i} A_ij*x[j]) / A_ii are race-free and
// can execute in parallel.  A forward pass (color 0→nColors-1) followed by a
// backward pass (color nColors-1→0) gives the symmetric-GS smoothing
// required by GAMG.
//
// At coarser GAMG levels (matrix too small to have been uploaded) the smoother
// falls back to CPU damped Jacobi (omega = 2/3).
//
// Float32 note: GS replaces x[i] outright each sweep, so float32 rounding does
// not accumulate across sweeps.  Convergence to 1e-6 absolute tolerance is
// achieved cleanly, unlike Jacobi-based approaches that accumulate corrections.

class MetalGCGSSmoother
:
    public lduMatrix::smoother
{
    scalar     omega_;    // fallback Jacobi damping (2/3); unused on GPU path
    bool       useGPU_;
    MetalSpMV* sparseMV_;

public:

    TypeName("MetalGCGS");

    MetalGCGSSmoother
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces
    );

    MetalGCGSSmoother(const MetalGCGSSmoother&) = delete;
    void operator=(const MetalGCGSSmoother&) = delete;
    virtual ~MetalGCGSSmoother() {}

    virtual void smooth
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt,
        const label nSweeps
    ) const;
};


} // End namespace Foam

#endif
