#include "MetalGCGSSmoother.H"

namespace Foam
{
    defineTypeNameAndDebug(MetalGCGSSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<MetalGCGSSmoother>
        addMetalGCGSSymMatrixSmootherToTable_;
}


Foam::MetalGCGSSmoother::MetalGCGSSmoother
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
)
:
    lduMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces
    ),
    omega_(2.0/3.0),
    useGPU_(false),
    sparseMV_(MetalSpMV::sharedCached())
{
    // GPU path active only at the fine GAMG level where the singleton matrix
    // has been uploaded (size match confirms same matrix).
    useGPU_ = sparseMV_->isReady() &&
               (sparseMV_->nRows() == static_cast<int>(matrix.diag().size()));
}


void Foam::MetalGCGSSmoother::smooth
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    const label n = psi.size();

    // GPU GCGS path: only at the fine level on a single rank.
    // In parallel each rank's matrix has processor interfaces; those halo
    // contributions are not in the uploaded CSR, so we skip GPU and let the
    // CPU Jacobi fallback call matrix_.Amul (which handles MPI comms).
    bool levelHasInterfaces = false;
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i)) { levelHasInterfaces = true; break; }
    }

    if (useGPU_ && sparseMV_->hasGCGS() && !levelHasInterfaces)
    {
        sparseMV_->smoothGCGS(psi.begin(), source.begin(), n, nSweeps);
        return;
    }

    // Parallel fallback: delegate to CPU GaussSeidel so GAMG convergence quality
    // is preserved.  In serial without GCGS (coarse levels), also use GS.
    if (levelHasInterfaces || !useGPU_)
    {
        GaussSeidelSmoother::smooth
        (
            fieldName_,
            psi,
            matrix_,
            source,
            interfaceBouCoeffs_,
            interfaces_,
            cmpt,
            nSweeps
        );
        return;
    }

    // Coarse GAMG level, single rank, no GPU upload: damped Jacobi (omega=2/3).
    const scalarField& diag = matrix_.diag();
    scalarField Ap(n);
    for (label sweep = 0; sweep < nSweeps; ++sweep)
    {
        matrix_.Amul(Ap, psi, interfaceBouCoeffs_, interfaces_, cmpt);

        scalar* __restrict__       psiPtr  = psi.begin();
        const scalar* __restrict__ srcPtr  = source.begin();
        const scalar* __restrict__ ApPtr   = Ap.begin();
        const scalar* __restrict__ diagPtr = diag.begin();

        for (label i = 0; i < n; ++i)
            psiPtr[i] += omega_ * (srcPtr[i] - ApPtr[i]) / diagPtr[i];
    }
}
