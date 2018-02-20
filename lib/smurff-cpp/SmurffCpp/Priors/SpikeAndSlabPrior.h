#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Priors/NormalOnePrior.h>

namespace smurff {

// Spike and slab prior
class SpikeAndSlabPrior : public NormalOnePrior 
{
public:
   // updated by every thread
   smurff::thread_vector<Eigen::MatrixXd> Zcol, W2col;

   // read-only during sampling
   Eigen::MatrixXd Zkeep;
   Eigen::ArrayXXd alpha, log_alpha;
   Eigen::ArrayXXd r, log_r;

   //-- hyper params
   const double prior_beta = 1; //for r
   const double prior_alpha_0 = 1.; //for alpha
   const double prior_beta_0 = 1.; //for alpha

public:
   SpikeAndSlabPrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode);
   virtual ~SpikeAndSlabPrior() {}
   void init() override;

   std::pair<double,double> sample_latent(int d, int k, const Eigen::MatrixXd& XX, const Eigen::VectorXd& yX) override;

   void update_prior() override;

   // mean value of Z
   std::ostream &status(std::ostream &os, std::string indent) const override;
};
}
