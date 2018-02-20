#include "NormalOnePrior.h"
#include <SmurffCpp/IO/MatrixIO.h>

using namespace smurff;
using namespace Eigen;

NormalOnePrior::NormalOnePrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode, std::string name)
   : ILatentPrior(trainTask, mode, name)
{

}

void NormalOnePrior::init()
{
   //does not look that there was such init previously
   ILatentPrior::init();

   const int K = num_latent();
   mu.resize(K);
   mu.setZero();

   Lambda.resize(K, K);
   Lambda.setIdentity();
   Lambda *= 10.;

   // parameters of Inv-Whishart distribution
   WI.resize(K, K);
   WI.setIdentity();
   mu0.resize(K);
   mu0.setZero();
   b0 = 2;
   df = K;
}

const Eigen::VectorXd NormalOnePrior::getMu(int n) const
{
   return mu;
}

void NormalOnePrior::update_prior()
{
    std::tie(mu, Lambda) = CondNormalWishart(num_cols(), getUUsum(), getUsum(), mu0, b0, WI, df);
}

void NormalOnePrior::sample_latent(int d)
{
   const int K = num_latent();

   MatrixXd XX = MatrixXd::Zero(K, K);
   VectorXd yX = VectorXd::Zero(K);

   data()->getMuLambda(model(), m_mode, d, yX, XX);

   // add hyperparams
   yX.noalias() += Lambda * mu;
   XX.noalias() += Lambda;

   for(int k=0;k<K;++k) sample_latent(d, k, XX, yX);
}
 
std::pair<double,double> NormalOnePrior::sample_latent(int d, int k, const MatrixXd& XX, const VectorXd& yX)
{
    auto Ucol = U().col(d);
    double lambda = XX(k,k);
    double mu = (1/lambda) * (yX(k) - Ucol.transpose() * XX.col(k) + Ucol(k) * XX(k,k));
    Ucol(k) = mu + randn() / sqrt(lambda);
    return std::make_pair(mu, lambda);
}

std::ostream &NormalOnePrior::status(std::ostream &os, std::string indent) const
{
   return os;
}
