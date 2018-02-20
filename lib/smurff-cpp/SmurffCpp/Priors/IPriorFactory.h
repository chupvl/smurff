#pragma once

#include <memory>

namespace smurff {
   
class TrainTask;
class ILatentPrior;

class IPriorFactory
{
public:
   virtual ~IPriorFactory()
   {
   }

public:
   virtual std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<TrainTask> trainTask, int mode) = 0;
};
   
}