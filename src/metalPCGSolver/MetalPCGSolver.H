#ifndef MetalPCGSolver_H
#define MetalPCGSolver_H

#include "lduMatrix.H"
#include "metalSpMV.h"
#include <vector>

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class MetalPCGSolver Declaration
\*---------------------------------------------------------------------------*/

class MetalPCGSolver
:
    public lduMatrix::solver
{
    // Pointer into the process-wide cached SpMV in MetalShared.
    // Not owned — the singleton owns the object.
    mutable MetalSpMV* sparseMV_;
    mutable bool       gpuReady_;

    // Convert the stored lduMatrix to CSR and upload to the GPU.
    // Called once at construction; assumes matrix coefficients do not
    // change between construction and solve() (true for one-shot solvers).
    void buildAndUploadCSR() const;

public:

    TypeName("metalPCG");

    MetalPCGSolver
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    MetalPCGSolver(const MetalPCGSolver&) = delete;
    virtual ~MetalPCGSolver() {}

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;

    void operator=(const MetalPCGSolver&) = delete;
};


} // End namespace Foam

#endif
