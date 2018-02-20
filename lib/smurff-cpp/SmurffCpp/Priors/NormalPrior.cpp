#include "NormalPrior.h"

#include <iomanip>

#include <SmurffCpp/Utils/chol.h>
#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/counters.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

using namespace Eigen;
using namespace smurff;

//  base class NormalPrior

NormalPrior::NormalPrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode, std::string name)
   : ILatentPrior(trainTask, mode, name)
{

}

void NormalPrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   const int K = num_latent();
   mu.resize(K);
   mu.setZero();

   Lambda.resize(K, K);
   Lambda.setIdentity();
   Lambda *= 10;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;
}

const Eigen::VectorXd NormalPrior::getMu(int n) const
{
   return mu;
}

void NormalPrior::update_prior()
{
   std::tie(mu, Lambda) = CondNormalWishart(num_cols(), getUUsum(), getUsum(), mu0, b0, WI, df);
}

//n is an index of column in U matrix
void  NormalPrior::sample_latent(int n)
{
   COUNTER("NormalPrior::sample_latent");
   const auto &mu_u = getMu(n);

   VectorXd &rr = rrs.local();
   MatrixXd &MM = MMs.local();

   rr.setZero();
   MM.setZero();

   // add pnm
   data()->getMuLambda(model(), m_mode, n, rr, MM);

   // add hyperparams
   rr.noalias() += Lambda * mu_u;
   MM.noalias() += Lambda;

   //Solve system of linear equations for x: MM * x = rr - not exactly correct  because we have random part
   //Sample from multivariate normal distribution with mean rr and precision matrix MM

   Eigen::LLT<MatrixXd> chol;
   {
      COUNTER("cholesky");
      chol = MM.llt(); // compute the Cholesky decomposition X = L * U
      if(chol.info() != Eigen::Success)
      {
         THROWERROR("Cholesky Decomposition failed!");
      }
   }

   chol.matrixL().solveInPlace(rr); // solve for y: y = L^-1 * b
   rr.noalias() += nrandn(num_latent());
   chol.matrixU().solveInPlace(rr); // solve for x: x = U^-1 * y
   
   U().col(n).noalias() = rr; // rr is equal to x
}

std::ostream &NormalPrior::status(std::ostream &os, std::string indent) const
{
   os << indent << m_name << ": mu = " <<  mu.norm() << std::endl;
   return os;
}
//macau Probit sample latents
/*
void BPMFPrior::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
   double mean_value, const Eigen::MatrixXd &samples, const int num_latent)
{
   const int N = U.cols();

   #pragma omp parallel for schedule(dynamic, 2)
   for(int n = 0; n < N; n++) 
   {
      sample_latent_blas_probit(U, n, mat, mean_value, samples, mu, Lambda, num_latent);
   }
}
*/

//macau Tensor methods
/*
void BPMFPrior::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, 
                               int mode, 
                               const int num_latent
                              ) 
{
  // TODO
  THROWERROR("Unimplemented: sample_latents");
}
*/
