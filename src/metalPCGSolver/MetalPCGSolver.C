#include "MetalPCGSolver.H"
#include <vector>
#include <algorithm>
#include <cmath>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(MetalPCGSolver, 0);

    // Register as a symmetric-matrix solver (same category as PCG)
    lduMatrix::solver::addsymMatrixConstructorToTable<MetalPCGSolver>
        addMetalPCGSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * Private Member Functions * * * * * * * * * * * //

void Foam::MetalPCGSolver::buildAndUploadCSR() const
{
    const scalarField& diag  = matrix_.diag();
    const scalarField& upper = matrix_.upper();
    const scalarField& lower = matrix_.lower();

    const labelUList& uAddr = matrix_.lduAddr().upperAddr();
    const labelUList& lAddr = matrix_.lduAddr().lowerAddr();

    const int nCells = diag.size();
    const int nFaces = upper.size();
    const int nnz    = nCells + 2 * nFaces;

    // Count non-zeros per row
    std::vector<int> rowCount(nCells, 1);  // 1 for diagonal
    for (int f = 0; f < nFaces; ++f)
    {
        rowCount[lAddr[f]]++;   // upper-triangle entry in row lAddr[f]
        rowCount[uAddr[f]]++;   // lower-triangle entry in row uAddr[f]
    }

    // Build rowPtr (prefix sum of rowCount)
    std::vector<int> rowPtr(nCells + 1, 0);
    for (int i = 0; i < nCells; ++i)
        rowPtr[i + 1] = rowPtr[i] + rowCount[i];

    std::vector<int>    colIdx(nnz);
    std::vector<double> values(nnz);

    // Track the current fill position for each row
    std::vector<int> pos(rowPtr.begin(), rowPtr.begin() + nCells);

    // Diagonal entries
    for (int i = 0; i < nCells; ++i)
    {
        colIdx[pos[i]] = i;
        values[pos[i]] = diag[i];
        pos[i]++;
    }

    // Off-diagonal entries
    // lAddr[f] < uAddr[f] by construction in OpenFOAM's LDU addressing.
    // upper[f] is the entry at (lAddr[f], uAddr[f]) — upper triangle.
    // lower[f] is the entry at (uAddr[f], lAddr[f]) — lower triangle.
    for (int f = 0; f < nFaces; ++f)
    {
        const int l = lAddr[f];
        const int u = uAddr[f];

        colIdx[pos[l]] = u;
        values[pos[l]] = upper[f];
        pos[l]++;

        colIdx[pos[u]] = l;
        values[pos[u]] = lower[f];
        pos[u]++;
    }

    const bool wasReady = sparseMV_->isReady();
    sparseMV_->setup(nCells, rowPtr.data(), colIdx.data(), values.data(), nnz);
    gpuReady_ = sparseMV_->isReady();

    if (gpuReady_ && !wasReady)
    {
        Info<< "metalPCG: GPU SpMV uploaded — "
            << nCells << " rows, " << nnz << " non-zeros" << endl;
    }
    else if (!gpuReady_)
    {
        Info<< "metalPCG: GPU setup failed, falling back to CPU" << endl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::MetalPCGSolver::MetalPCGSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    sparseMV_(MetalSpMV::sharedCached()),
    gpuReady_(false)
{
    buildAndUploadCSR();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::MetalPCGSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();
    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    // Check whether any processor/cyclic interface contributions exist.
    // If so, fall back to the CPU path (initMatrixInterfaces /
    // updateMatrixInterfaces handle those boundary contributions).
    bool hasInterfaces = false;
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            hasInterfaces = true;
            break;
        }
    }

    // Lambda that performs A*x → result, either on GPU or CPU
    auto amul = [&](scalarField& result, const scalarField& vec)
    {
        if (gpuReady_ && !hasInterfaces)
        {
            sparseMV_->multiply(result.begin(), vec.begin(), nCells);
        }
        else
        {
            matrix_.Amul(result, vec, interfaceBouCoeffs_, interfaces_, cmpt);
        }
    };

    scalar wArA    = solverPerf.great_;
    scalar wArAold = wArA;

    // Initial residual: wA = A*psi, rA = source - wA
    amul(wA, psi);
    scalarField rA(source - wA);
    scalar* __restrict__ rAPtr = rA.begin();

    scalar normFactor = this->normFactor(psi, source, wA, pA);

    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm()) / normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if (minIter_ > 0 || !solverPerf.checkConvergence(tolerance_, relTol_))
    {
        autoPtr<lduMatrix::preconditioner> preconPtr =
            lduMatrix::preconditioner::New(*this, controlDict_);

        do
        {
            wArAold = wArA;

            preconPtr->precondition(wA, rA, cmpt);

            wArA = gSumProd(wA, rA, matrix().mesh().comm());

            if (solverPerf.nIterations() == 0)
            {
                for (label cell = 0; cell < nCells; cell++)
                    pAPtr[cell] = wAPtr[cell];
            }
            else
            {
                scalar beta = wArA / wArAold;
                for (label cell = 0; cell < nCells; cell++)
                    pAPtr[cell] = wAPtr[cell] + beta * pAPtr[cell];
            }

            amul(wA, pA);

            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());

            if (solverPerf.checkSingularity(mag(wApA) / normFactor)) break;

            scalar alpha = wArA / wApA;

            for (label cell = 0; cell < nCells; cell++)
            {
                psiPtr[cell] += alpha * pAPtr[cell];
                rAPtr[cell]  -= alpha * wAPtr[cell];
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm()) / normFactor;

        } while
        (
            (
               ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    return solverPerf;
}
