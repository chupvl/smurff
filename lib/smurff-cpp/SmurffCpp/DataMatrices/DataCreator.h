#pragma once

#include <memory>

#include "IDataCreator.h"

#include <SmurffCpp/Sessions/Session.h>

namespace smurff
{
   class DataCreator : public IDataCreator
   {
   private:
      std::shared_ptr<TrainTask> m_trainTask;

   public:
      DataCreator(std::shared_ptr<TrainTask> trainTask)
         : m_trainTask(trainTask)
      {
      }

   public:
      std::shared_ptr<Data> create(std::shared_ptr<const MatrixConfig> mc) const override;
      std::shared_ptr<Data> create(std::shared_ptr<const TensorConfig> tc) const override;
   };
}
