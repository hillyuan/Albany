#include "Albany_CombineAndScatterManagerTpetra.hpp"

#include "Albany_TpetraThyraTypes.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace {
Tpetra::CombineMode combineModeT (const Albany::CombineMode modeA)
{
  Tpetra::CombineMode modeT;
  switch (modeA) {
    case Albany::CombineMode::ADD:
      modeT = Tpetra::CombineMode::ADD;
      break;
    case Albany::CombineMode::INSERT:
      modeT = Tpetra::CombineMode::INSERT;
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unknown Albany combine mode. Please, contact developers.\n");
  }
  return modeT;
}

} // anonymous namespace

namespace Albany
{

CombineAndScatterManagerTpetra::
CombineAndScatterManagerTpetra(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                               const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
 : owned_vs      (owned)
 , overlapped_vs (overlapped)
{
  auto ownedT = Albany::getTpetraMap(owned);
  auto overlappedT = Albany::getTpetraMap(overlapped);

  importer = Teuchos::rcp( new Tpetra_Import(ownedT, overlappedT) );
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<Thyra_Vector>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraVector(src);
  auto dstT = Albany::getTpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's source map.\n");
#endif

  dstT->doExport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMultiVector(src);
  auto dstT = Albany::getTpetraMultiVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's source map.\n");
#endif

  dstT->doExport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMatrix(src);
  auto dstT = Albany::getTpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstT->doExport(*srcT,*importer,cmT);
}

// Scatter methods
void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<Thyra_Vector>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraVector(src);
  auto dstT = Albany::getTpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's target map.\n");
#endif

  dstT->doImport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
              const Teuchos::RCP<Thyra_MultiVector>& dst,
              const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMultiVector(src);
  auto dstT = Albany::getTpetraMultiVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's target map.\n");
#endif

  dstT->doImport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
              const Teuchos::RCP<Thyra_LinearOp>& dst,
              const CombineMode CM) const
{
  auto cmT  = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMatrix(src);
  auto dstT = Albany::getTpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstT->doImport(*srcT,*importer,cmT);
}

} // namespace Albany