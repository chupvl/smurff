#include "DataCreator.h"

#include "DataCreatorBase.h"

#include <SmurffCpp/DataMatrices/MatricesData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/Utils/PVec.hpp>

using namespace smurff;

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const MatrixConfig> mc) const
{
   //row_matrices and col_matrices are selected if prior is not macau and not macauone
   std::vector<std::shared_ptr<TensorConfig> > row_matrices;
   std::vector<std::shared_ptr<TensorConfig> > col_matrices;

   //row prior
   PriorTypes rowPriorType = m_trainTask->getConfig().getPriorTypes().at(0);
   if (rowPriorType != PriorTypes::macau && rowPriorType != PriorTypes::macauone)
      row_matrices = m_trainTask->getConfig().getAuxData().at(0);

   //col prior
   PriorTypes colPriorType = m_trainTask->getConfig().getPriorTypes().at(1);
   if (colPriorType != PriorTypes::macau && colPriorType != PriorTypes::macauone)
      col_matrices = m_trainTask->getConfig().getAuxData().at(1);

   //create creator
   std::shared_ptr<DataCreatorBase> creatorBase = std::make_shared<DataCreatorBase>();

   //create single matrix
   if (row_matrices.empty() && col_matrices.empty())
      return mc->create(creatorBase);

   //multiple matrices
   NoiseConfig ncfg(NoiseTypes::unused);
   std::shared_ptr<MatricesData> local_data_ptr(new MatricesData());
   local_data_ptr->setNoiseModel(NoiseFactory::create_noise_model(ncfg));
   local_data_ptr->add(PVec<>({0,0}), mc->create(creatorBase));

   for(size_t i = 0; i < row_matrices.size(); ++i)
   {
      local_data_ptr->add(PVec<>({0, static_cast<int>(i + 1)}), row_matrices[i]->create(creatorBase));
   }

   for(size_t i = 0; i < col_matrices.size(); ++i)
   {
      local_data_ptr->add(PVec<>({static_cast<int>(i + 1), 0}), col_matrices[i]->create(creatorBase));
   }

   return local_data_ptr;
}

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const TensorConfig> tc) const
{
   for (const auto& sideInfo : m_trainTask->getConfig().getSideInfo())
   {
      if (sideInfo)
      {
         THROWERROR("Tensor config does not support side info");
      }
   }

   for (const auto& auxDataSet : m_trainTask->getConfig().getAuxData())
   {
      if (!auxDataSet.empty())
      {
         THROWERROR("Tensor config does not support aux data");
      }
   }

   //create creator
   std::shared_ptr<DataCreatorBase> creatorBase = std::make_shared<DataCreatorBase>();

   return tc->create(creatorBase);
}
