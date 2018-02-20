#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/ILatentPrior.h>

namespace smurff {

// Prior without side information (pure BPMF) 
class NormalPrior : public ILatentPrior 
{
public:
  // hyperparams
  Eigen::VectorXd mu; 
  Eigen::MatrixXd Lambda;
  Eigen::MatrixXd WI;
  Eigen::VectorXd mu0;

  // constants
  int b0;
  int df;

protected:
   NormalPrior()
      : ILatentPrior(){}

public:
  NormalPrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode, std::string name = "NormalPrior");
  virtual ~NormalPrior() {}
  void init() override;

  //mu in NormalPrior does not depend on column index
  //however successors of this class can override this method
  //for example in MacauPrior mu depends on Uhat.col(n)
  virtual const Eigen::VectorXd getMu(int n) const;
  
  void sample_latent(int n) override;

  void update_prior() override;
  std::ostream &status(std::ostream &os, std::string indent) const override;

private:
  void initUU();
};
}
