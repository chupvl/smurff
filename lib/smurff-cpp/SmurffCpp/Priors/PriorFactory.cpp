#include "PriorFactory.h"

#include <Eigen/Core>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/Priors/NormalOnePrior.h>
#include <SmurffCpp/Priors/SpikeAndSlabPrior.h>

#include <SmurffCpp/Utils/Error.h>

using namespace smurff;
using namespace Eigen;

//create macau prior features

std::shared_ptr<Eigen::MatrixXd> PriorFactory::side_info_config_to_dense_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   Eigen::MatrixXd sideinfo = matrix_utils::dense_to_eigen(*sideinfoConfig);
   return std::shared_ptr<Eigen::MatrixXd>(new Eigen::MatrixXd(sideinfo));
}

std::shared_ptr<SparseFeat> PriorFactory::side_info_config_to_sparse_binary_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig->getNRow();
   std::uint64_t ncol = sideinfoConfig->getNCol();
   std::uint64_t nnz = sideinfoConfig->getNNZ();

   std::shared_ptr<std::vector<std::uint32_t> > rows = sideinfoConfig->getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > cols = sideinfoConfig->getColsPtr();

   // Temporary solution. As soon as SparseFeat works with vectors instead of pointers,
   // we will remove these extra memory allocation and manipulation
   int* rowsRawPtr = new int[nnz];
   int* colsRawPtr = new int[nnz];
   for (std::uint64_t i = 0; i < nnz; i++)
   {
      rowsRawPtr[i] = rows->operator[](i);
      colsRawPtr[i] = cols->operator[](i);
   }

   return std::shared_ptr<SparseFeat>(new SparseFeat(nrow, ncol, nnz, rowsRawPtr, colsRawPtr));
}

std::shared_ptr<SparseDoubleFeat> PriorFactory::side_info_config_to_sparse_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode)
{
   std::uint64_t nrow = sideinfoConfig->getNRow();
   std::uint64_t ncol = sideinfoConfig->getNCol();
   std::uint64_t nnz = sideinfoConfig->getNNZ();

   std::shared_ptr<std::vector<std::uint32_t> > rows = sideinfoConfig->getRowsPtr();
   std::shared_ptr<std::vector<std::uint32_t> > cols = sideinfoConfig->getColsPtr();
   std::shared_ptr<std::vector<double> > values = sideinfoConfig->getValuesPtr();

   // Temporary solution. As soon as SparseDoubleFeat works with vectorsor shared pointers instead of raw pointers,
   // we will remove these extra memory allocation and manipulation
   int* rowsRawPtr = new int[nnz];
   int* colsRawPtr = new int[nnz];
   double* valuesRawPtr = new double[nnz];
   for (size_t i = 0; i < nnz; i++)
   {
      rowsRawPtr[i] = rows->operator[](i);
      colsRawPtr[i] = cols->operator[](i);
      valuesRawPtr[i] = values->operator[](i);
   }

   return std::shared_ptr<SparseDoubleFeat>(new SparseDoubleFeat(nrow, ncol, nnz, rowsRawPtr, colsRawPtr, valuesRawPtr));
}

//-------

std::shared_ptr<ILatentPrior> PriorFactory::create_prior(std::shared_ptr<TrainTask> trainTask, int mode)
{
   PriorTypes priorType = trainTask->getConfig().getPriorTypes().at(mode);

   switch(priorType)
   {
   case PriorTypes::normal:
   case PriorTypes::default_prior:
      return std::shared_ptr<NormalPrior>(new NormalPrior(trainTask, -1));
   case PriorTypes::spikeandslab:
      return std::shared_ptr<SpikeAndSlabPrior>(new SpikeAndSlabPrior(trainTask, -1));
   case PriorTypes::normalone:
      return std::shared_ptr<NormalOnePrior>(new NormalOnePrior(trainTask, -1));
   case PriorTypes::macau:
   case PriorTypes::macauone:
      return create_macau_prior<PriorFactory>(trainTask, mode, priorType, trainTask->getConfig().getSideInfo().at(mode));
   default:
      {
         THROWERROR("Unknown prior: " + priorTypeToString(priorType));
      }
   }
}
