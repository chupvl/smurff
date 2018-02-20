#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/DataMatrices/ScarceMatrixData.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>

#include <SmurffCpp/Priors/NormalOnePrior.h>

namespace smurff {

//Why remove init method and put everything in constructor if we have
//init method in other priors and the other method addSideInfo which we use in pair

template<class FType>
class MacauOnePrior : public NormalOnePrior
{
public:
   typedef FType SideInfo;

   Eigen::MatrixXd Uhat;

   std::shared_ptr<FType> Features;  // side information
   Eigen::VectorXd F_colsq;   // sum-of-squares for every feature (column)

   Eigen::MatrixXd beta;      // link matrix
   double lb0 = 5.0;
   Eigen::VectorXd lambda_beta;
   double lambda_beta_a0; // Hyper-prior for lambda_beta
   double lambda_beta_b0; // Hyper-prior for lambda_beta

public:
   MacauOnePrior(std::shared_ptr<TrainTask> session, uint32_t mode)
      : NormalOnePrior(session, mode, "MacauOnePrior")
   {
   }

   void init() override
   {
      NormalOnePrior::init();

      // init SideInfo related
      Uhat = Eigen::MatrixXd::Constant(num_latent(), Features->rows(), 0.0);
      beta = Eigen::MatrixXd::Constant(num_latent(), Features->cols(), 0.0);

      // initial value (should be determined automatically)
      // Hyper-prior for lambda_beta (mean 1.0):
      lambda_beta     = Eigen::VectorXd::Constant(num_latent(), lb0);
      lambda_beta_a0 = 0.1;
      lambda_beta_b0 = 0.1;
   }

   void addSideInfo(std::shared_ptr<FType> &Fmat, bool)
   {
      // side information
      Features = Fmat;
      F_colsq = smurff::linop::col_square_sum(*Features);
   }

   void update_prior() override
   {
      sample_mu_lambda(U());
      sample_beta(U());
      smurff::linop::compute_uhat(Uhat, *Features, beta);
      sample_lambda_beta();
   }
    
   const Eigen::VectorXd getMu(int n) const override
   {
      return this->mu + Uhat.col(n);
   }


   double getLinkLambda()
   {
      return lambda_beta.mean();
   }

   //used in update_prior

   void sample_beta(const Eigen::MatrixXd &U)
   {
      // updating beta and beta_var
      const int nfeat = beta.cols();
      const int N = U.cols();
      const int blocksize = 4;

      Eigen::MatrixXd Z;

      #pragma omp parallel for private(Z) schedule(static, 1)
      for (int dstart = 0; dstart < num_latent(); dstart += blocksize)
      {
         const int dcount = std::min(blocksize, num_latent() - dstart);
         Z.resize(dcount, U.cols());

         for (int i = 0; i < N; i++)
         {
            for (int d = 0; d < dcount; d++)
            {
               int dx = d + dstart;
               Z(d, i) = U(dx, i) - mu(dx) - Uhat(dx, i);
            }
         }

         for (int f = 0; f < nfeat; f++)
         {
            Eigen::VectorXd zx(dcount), delta_beta(dcount), randvals(dcount);
            // zx = Z[dstart : dstart + dcount, :] * F[:, f]
            smurff::linop::At_mul_Bt(zx, *Features, f, Z);
            // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
            bmrandn_single( randvals );

            for (int d = 0; d < dcount; d++)
            {
               int dx = d + dstart;
               double A_df     = lambda_beta(dx) + Lambda(dx,dx) * F_colsq(f);
               double B_df     = Lambda(dx,dx) * (zx(d) + beta(dx,f) * F_colsq(f));
               double A_inv    = 1.0 / A_df;
               double beta_new = B_df * A_inv + std::sqrt(A_inv) * randvals(d);
               delta_beta(d)   = beta(dx,f) - beta_new;

               beta(dx, f)     = beta_new;
            }
            // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
            smurff::linop::add_Acol_mul_bt(Z, *Features, f, delta_beta);
         }
      }
   }

   //used in update_prior

   void sample_mu_lambda(const Eigen::MatrixXd &U)
   {
      Eigen::MatrixXd WI(num_latent(), num_latent());
      WI.setIdentity();
      int N = U.cols();

      Eigen::MatrixXd Udelta(num_latent(), N);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < N; i++)
      {
         for (int d = 0; d < num_latent(); d++)
         {
            Udelta(d, i) = U(d, i) - Uhat(d, i);
         }
      }
      std::tie(mu, Lambda) = CondNormalWishart(Udelta, Eigen::VectorXd::Constant(num_latent(), 0.0), 2.0, WI, num_latent());
   }

   //used in update_prior

   void sample_lambda_beta()
   {
      double lambda_beta_a = lambda_beta_a0 + beta.cols() / 2.0;
      Eigen::VectorXd lambda_beta_b = Eigen::VectorXd::Constant(beta.rows(), lambda_beta_b0);
      const int D = beta.rows();
      const int F = beta.cols();
      #pragma omp parallel
      {
         Eigen::VectorXd tmp(D);
         tmp.setZero();
         #pragma omp for schedule(static)
         for (int f = 0; f < F; f++)
         {
            for (int d = 0; d < D; d++)
            {
               tmp(d) += std::pow(beta(d, f), 2);
            }
         }
         #pragma omp critical
         {
            lambda_beta_b += tmp / 2;
         }
      }
      for (int d = 0; d < D; d++)
      {
         lambda_beta(d) = rgamma(lambda_beta_a, 1.0 / lambda_beta_b(d));
      }
   }

   void setLambdaBeta(double lb)
   {
       lb0 = lb;
   }

   void setTol(double)
   {
   }

   void save(std::shared_ptr<const StepFile> sf) const override
   {
      NormalOnePrior::save(sf);

      std::string path = sf->getPriorFileName(m_mode);
      smurff::matrix_io::eigen::write_matrix(path, beta);
   }

   void restore(std::shared_ptr<const StepFile> sf) override
   {
      NormalOnePrior::restore(sf);

      std::string path = sf->getPriorFileName(m_mode);

      THROWERROR_FILE_NOT_EXIST(path);

      smurff::matrix_io::eigen::read_matrix(path, beta);
   }

   std::ostream &status(std::ostream &os, std::string indent) const override
   {
      os << indent << "  " << m_name << ": Beta = " << beta.norm() << "\n";
      return os;
   }
};

}

//==========================

//macau Probit sample latents
/*
template<class FType>
void MacauOnePrior<FType>::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                          double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
 //TODO
 THROWERROR("Not implemented!");
}
*/

//macau Tensor method
/*
template<class FType>
void MacauOnePrior<FType>::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                          std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent)
{
  THROWERROR("Unimplemented: sample_latents");
}
*/

//macau Tensor method
/*
template<class FType>
void MacauOnePrior<FType>::sample_latents(double noisePrecision, TensorData & data,
                                          std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent)
{
  auto& sparseMode = (*data.Y)[mode];
  auto& U = samples[mode];
  const int N = U->cols();
  const int D = num_latent;
  VectorView<Eigen::MatrixXd> view(samples, mode);
  const int nmodes1 = view.size();
  const double mean_value = data.mean_value;

  if (U->rows() != num_latent)
  {
    THROWERROR("U->rows() must be equal to num_latent.");
  }

  Eigen::VectorXi & row_ptr = sparseMode->row_ptr;
  Eigen::MatrixXi & indices = sparseMode->indices;
  Eigen::VectorXd & values  = sparseMode->values;

  #pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < N; i++) {
    // looping over all non-zeros for row i of the mode
    // precalculating Yhat and Qi
    const int nnz = row_ptr(i + 1) - row_ptr(i);
    VectorXd Yhat(nnz);
    VectorXd tmpd(nnz);
    VectorXd Qi = lambda;

    for (int idx = 0; idx < nnz; idx++) {
      int j = idx + row_ptr(i);
      VectorXd prod = VectorXd::Ones(D);
      for (int m = 0; m < nmodes1; m++) {
        auto v = view.get(m)->col(indices(j, m));
        prod.noalias() = prod.cwiseProduct(v);
      }
      Qi.noalias() += noisePrecision * prod.cwiseAbs2();
      Yhat(idx) = mean_value + U->col(i).dot(prod);
    }

    // generating random numbers
    VectorXd rnorms(num_latent);
    bmrandn_single(rnorms);

    for (int d = 0; d < D; d++)
    {
      // computing Lid
      const double uid = (*U)(d, i);
      double Lid = lambda(d) * (mu(d) + Uhat(d, i));

      for (int idx = 0; idx < nnz; idx++)
      {
        int j = idx + row_ptr(i);

        // computing t = vjd * wkd * ..
        double t = 1.0;
        for (int m = 0; m < nmodes1; m++)
        {
          t *= (*view.get(m))(d, indices(j, m));
        }
        tmpd(idx) = t;
        // L_id += alpha * (Y_ijk - k_ijkd) * v_jd * wkd
        Lid += noisePrecision * (values(j) - (Yhat(idx) - uid * t)) * t;
      }
      // Now use Lid and Qid to update uid
      double uid_old = uid;
      double uid_var = 1.0 / Qi(d);

      // sampling new u_id ~ Norm(Lid / Qid, 1/Qid)
      (*U)(d, i) = Lid * uid_var + std::sqrt(uid_var) * rnorms(d);

      // updating Yhat
      double uid_delta = (*U)(d, i) - uid_old;
      for (int idx = 0; idx < nnz; idx++) {
        Yhat(idx) += uid_delta * tmpd(idx);
      }
    }
  }
}
*/
