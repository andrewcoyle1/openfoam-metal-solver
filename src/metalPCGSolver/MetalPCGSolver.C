#include "MetalPCGSolver.H"
#include <vector>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(MetalPCGSolver, 0);

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

    std::vector<int> rowCount(nCells, 1);
    for (int f = 0; f < nFaces; ++f)
    {
        rowCount[lAddr[f]]++;
        rowCount[uAddr[f]]++;
    }

    std::vector<int> rowPtr(nCells + 1, 0);
    for (int i = 0; i < nCells; ++i)
        rowPtr[i + 1] = rowPtr[i] + rowCount[i];

    std::vector<int>    colIdx(nnz);
    std::vector<double> values(nnz);

    std::vector<int> pos(rowPtr.begin(), rowPtr.begin() + nCells);

    for (int i = 0; i < nCells; ++i)
    {
        colIdx[pos[i]] = i;
        values[pos[i]] = diag[i];
        pos[i]++;
    }

    for (int f = 0; f < nFaces; ++f)
    {
        const int l = lAddr[f];
        const int u = uAddr[f];
        colIdx[pos[l]] = u;  values[pos[l]] = upper[f];  pos[l]++;
        colIdx[pos[u]] = l;  values[pos[u]] = lower[f];  pos[u]++;
    }

    const bool wasReady = sparseMV_->isReady();
    sparseMV_->setup(nCells, rowPtr.data(), colIdx.data(), values.data(), nnz);
    gpuReady_ = sparseMV_->isReady();

    if (gpuReady_ && !wasReady)
        Info<< "metalPCG: GPU SpMV uploaded — "
            << nCells << " rows, " << nnz << " non-zeros"
            << " (rank " << Pstream::myProcNo() << ")" << endl;
    else if (!gpuReady_)
        Info<< "metalPCG: GPU setup failed, falling back to CPU" << endl;
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

    bool hasInterfaces = false;
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i)) { hasInterfaces = true; break; }
    }

    // Temporary CPU vectors used by both paths
    scalarField pA(nCells, 0);
    scalarField wA(nCells, 0);

    // Hybrid Amul: GPU local SpMV + CPU halo exchange.
    // initMatrixInterfaces posts non-blocking MPI sends/receives; GPU SpMV runs
    // concurrently; updateMatrixInterfaces waits and adds halo contributions.
    auto hybridAmul = [&](scalarField& result, const scalarField& vec)
    {
        if (gpuReady_ && hasInterfaces)
        {
            matrix_.initMatrixInterfaces
                (interfaceBouCoeffs_, interfaces_, vec, result, cmpt);
            sparseMV_->multiply(result.begin(), vec.begin(), nCells);
            matrix_.updateMatrixInterfaces
                (interfaceBouCoeffs_, interfaces_, vec, result, cmpt);
        }
        else if (gpuReady_)
        {
            sparseMV_->multiply(result.begin(), vec.begin(), nCells);
        }
        else
        {
            matrix_.Amul(result, vec, interfaceBouCoeffs_, interfaces_, cmpt);
        }
    };

    // Compute A*psi for normFactor.
    hybridAmul(wA, psi);

    scalarField rA(source - wA);
    scalar normFactor = this->normFactor(psi, source, wA, pA);

    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm()) / normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if (!(minIter_ > 0 || !solverPerf.checkConvergence(tolerance_, relTol_)))
        return solverPerf;

    autoPtr<lduMatrix::preconditioner> preconPtr =
        lduMatrix::preconditioner::New(*this, controlDict_);

    // -----------------------------------------------------------------------
    // GPU PCG path — full PCG loop with GPU dot products, SAXPY, and SpMV.
    // Falls back to CPU loop on failure or when interfaces are present.
    // -----------------------------------------------------------------------
    if (gpuReady_ && !hasInterfaces)
    {
        scalarField rTmp(nCells), wTmp(nCells);

        auto precondFn = [&](double* w, const double* r, int n)
        {
            for (int i = 0; i < n; ++i) rTmp[i] = r[i];
            preconPtr->precondition(wTmp, rTmp, cmpt);
            for (int i = 0; i < n; ++i) w[i] = wTmp[i];
        };

        double initRes, finalRes;
        int nIter;

        bool ok = sparseMV_->solveGPUPCG(
            psi.begin(),
            source.begin(),
            nCells,
            normFactor,
            tolerance_,
            relTol_,
            maxIter_,
            minIter_,
            precondFn,
            initRes,
            finalRes,
            nIter
        );

        if (ok)
        {
            solverPerf.initialResidual() = initRes;
            solverPerf.finalResidual()   = finalRes;
            solverPerf.nIterations()     = nIter;
            return solverPerf;
        }
    }

    // -----------------------------------------------------------------------
    // CPU fallback path
    // -----------------------------------------------------------------------
    scalar* __restrict__ psiPtr = psi.begin();
    scalar* __restrict__ pAPtr  = pA.begin();
    scalar* __restrict__ wAPtr  = wA.begin();
    scalar* __restrict__ rAPtr  = rA.begin();

    // CPU PCG loop reuses hybridAmul defined above.

    scalar wArA    = solverPerf.great_;
    scalar wArAold = wArA;

    // Recompute initial residual from scratch on CPU path
    hybridAmul(wA, psi);
    for (label cell = 0; cell < nCells; cell++)
        rAPtr[cell] = source[cell] - wAPtr[cell];

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

        hybridAmul(wA, pA);

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

    return solverPerf;
}
