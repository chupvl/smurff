#pragma once

#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Priors/ILatentPrior.h>
#include <SmurffCpp/Priors/MacauOnePrior.hpp>
#include <SmurffCpp/Priors/MacauPrior.hpp>
#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Configs/MatrixConfig.h>

namespace smurff {

class PriorFactory : public IPriorFactory
{
private:

    std::shared_ptr<Eigen::MatrixXd> side_info_config_to_dense_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode);
    std::shared_ptr<SparseFeat> side_info_config_to_sparse_binary_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode);
    std::shared_ptr<SparseDoubleFeat> side_info_config_to_sparse_features(std::shared_ptr<MatrixConfig> sideinfoConfig, int mode);

public:
    template<class MacauPrior>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<TrainTask> trainTask, std::shared_ptr<typename MacauPrior::SideInfo> side_info);

    template<class SideInfo>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<TrainTask> trainTask, PriorTypes prior_type, std::shared_ptr<SideInfo> side_info);

    template<class Factory>
    std::shared_ptr<ILatentPrior> create_macau_prior(std::shared_ptr<TrainTask> trainTask, int mode, PriorTypes prior_type, const std::shared_ptr<MatrixConfig>& sideinfoConfig);

    std::shared_ptr<ILatentPrior> create_prior(std::shared_ptr<TrainTask> trainTask, int mode) override;
};

//-------

template<class MacauPrior>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<TrainTask> trainTask, std::shared_ptr<typename MacauPrior::SideInfo> side_info)
{
    std::shared_ptr<MacauPrior> prior(new MacauPrior(trainTask, -1));

    prior->addSideInfo(side_info, trainTask->getConfig().getDirect());
    prior->setLambdaBeta(trainTask->getConfig().getLambdaBeta());
    prior->setTol(trainTask->getConfig().getTol());

    return prior;
}

template<class SideInfo>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<TrainTask> trainTask, PriorTypes prior_type, std::shared_ptr<SideInfo> side_info)
{
   if(prior_type == PriorTypes::macau || prior_type == PriorTypes::default_prior)
   {
      return create_macau_prior<MacauPrior<SideInfo>>(trainTask, side_info);
   }
   else if(prior_type == PriorTypes::macauone)
   {
      return create_macau_prior<MacauOnePrior<SideInfo>>(trainTask, side_info);
   }
   else
   {
      THROWERROR("Unknown prior with side info: " + priorTypeToString(prior_type));
   }
}

//mode - 0 (row), 1 (col)
//vsideinfo - vector of side feature configs (row or col)
template<class Factory>
std::shared_ptr<ILatentPrior> PriorFactory::create_macau_prior(std::shared_ptr<TrainTask> trainTask, int mode, PriorTypes prior_type, const std::shared_ptr<MatrixConfig>& sideinfoConfig)
{
   Factory &subFactory = dynamic_cast<Factory &>(*this);

   if(!sideinfoConfig)
   {
      THROWERROR("Side info should always present for macau prior");
   }

   if (sideinfoConfig->isBinary())
   {
      std::shared_ptr<SparseFeat> sideinfo = side_info_config_to_sparse_binary_features(sideinfoConfig, mode);
      return subFactory.create_macau_prior(trainTask, prior_type, sideinfo);
   }
   else if (sideinfoConfig->isDense())
   {
      std::shared_ptr<Eigen::MatrixXd> sideinfo = side_info_config_to_dense_features(sideinfoConfig, mode);
      return subFactory.create_macau_prior(trainTask, prior_type, sideinfo);
   }
   else
   {
      std::shared_ptr<SparseDoubleFeat> sideinfo = side_info_config_to_sparse_features(sideinfoConfig, mode);
      return subFactory.create_macau_prior(trainTask, prior_type, sideinfo);
   }
}
}
