#include "SpikeAndSlabPrior.h"

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/Utils/Error.h>

using namespace smurff;
using namespace Eigen;

SpikeAndSlabPrior::SpikeAndSlabPrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode)
   : NormalOnePrior(trainTask, mode, "SpikeAndSlabPrior")
{

}

void SpikeAndSlabPrior::init()
{
   NormalOnePrior::init();

   const int K = num_latent();
   const int D = num_cols();
   const int nview = data()->nview(m_mode);
   
   THROWERROR_ASSERT(D > 0);

   Zcol.init(MatrixXd::Zero(K,nview));
   W2col.init(MatrixXd::Zero(K,nview));

   //-- prior params
   alpha = ArrayXXd::Ones(K,nview);
   Zkeep = MatrixXd::Constant(K, nview, D);
   r = ArrayXXd::Constant(K,nview,.5);

   // derived prior parameters
   log_alpha = alpha.log();
   log_r = - r.log() + (ArrayXXd::Ones(K, nview) - r).log();

}

void SpikeAndSlabPrior::update_prior()
{
   const int K = num_latent();
   const int nview = data()->nview(m_mode);
   
   auto Zc = Zcol.combine();
   auto W2c = W2col.combine();

   // update hyper params (per view)
   for(int v=0; v<nview; ++v) {
       const int D = data()->view_size(m_mode, v);
       r.col(v) = ( Zc.col(v).array() + prior_beta ) / ( D + prior_beta * D ) ;
       auto ww = W2c.col(v).array() / 2 + prior_beta_0;
       auto tmpz = Zc.col(v).array() / 2 + prior_alpha_0 ;
       alpha.col(v) = tmpz.binaryExpr(ww, [](double a, double b)->double {
               return rgamma(a, 1/b) + 1e-7;
       });
   }

   log_alpha = alpha.log();
   log_r = - r.log() + (ArrayXXd::Ones(K, nview) - r).log();

   Zkeep = Zc.array();
   Zcol.reset();
   W2col.reset();
}

std::pair<double, double> SpikeAndSlabPrior::sample_latent(int d, int k, const MatrixXd& XX, const VectorXd& yX)
{
    const int v = data()->view(m_mode, d);
    double mu, lambda;

    MatrixXd aXX = alpha.matrix().col(v).asDiagonal();
    aXX += XX;
    std::tie(mu, lambda) = NormalOnePrior::sample_latent(d, k, aXX, yX);

    auto Ucol = U().col(d);
    double z1 = log_r(k,v) -  0.5 * (lambda * mu * mu - std::log(lambda) + log_alpha(k,v));
    double z = 1 / (1 + exp(z1));
    double p = rand_unif(0,1);
    if (Zkeep(k,v) > 0 && p < z) {
        Zcol.local()(k,v)++;
        W2col.local()(k,v) += Ucol(k) * Ucol(k);
    } else {
        Ucol(k) = .0;
    }

    return std::make_pair(mu, lambda);
}

std::ostream &SpikeAndSlabPrior::status(std::ostream &os, std::string indent) const
{
   const int V = data()->nview(m_mode);
   for(int v=0; v<V; ++v) 
   {
       int Zcount = (Zkeep.col(v).array() > 0).count();
       os << indent << m_name << ": Z[" << v << "] = " << Zcount << "/" << num_latent() << "\n";
   }
   return os;
}
