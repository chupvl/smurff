#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/IO/MatrixIO.h>
#include <SmurffCpp/IO/GenericIO.h>

#include <SmurffCpp/Utils/linop.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/Priors/NormalPrior.h>
#include <SmurffCpp/SparseDoubleFeat.h>
#include <SmurffCpp/SparseFeat.h>

namespace smurff {

//sample_beta method is now virtual. because we override it in MPIMacauPrior
//we also have this method in MacauOnePrior but it is not virtual
//maybe make it virtual?

/// Prior with side information
template<class FType>
class MacauPrior : public NormalPrior
{
public:
   typedef FType SideInfo;

   Eigen::MatrixXd Uhat;
   std::shared_ptr<FType> Features;  // side information
   Eigen::MatrixXd FtF;       // F'F
   Eigen::MatrixXd beta;      // link matrix
   bool use_FtF;
   double lambda_beta;
   double lambda_beta_mu0; // Hyper-prior for lambda_beta
   double lambda_beta_nu0; // Hyper-prior for lambda_beta

   double tol = 1e-6;

private:
   MacauPrior()
      : NormalPrior(){}

public:
   MacauPrior(std::shared_ptr<TrainTask> session, uint32_t mode)
      : NormalPrior(session, mode, "MacauPrior")
   {

   }

   virtual ~MacauPrior() {}

   void init() override
   {
      NormalPrior::init();

      THROWERROR_ASSERT_MSG(Features->rows() == num_cols(), "Number of rows in train must be equal to number of rows in features");

      if (use_FtF)
      {
         FtF.resize(Features->cols(), Features->cols());
         smurff::linop::At_mul_A(FtF, *Features);
      }

      Uhat.resize(this->num_latent(), Features->rows());
      Uhat.setZero();

      beta.resize(this->num_latent(), Features->cols());
      beta.setZero();
   }

   void update_prior() override
   {
      // residual (Uhat is later overwritten):
      Uhat.noalias() = U() - Uhat;
      Eigen::MatrixXd BBt = smurff::linop::A_mul_At_combo(beta);

      // sampling Gaussian
      std::tie(this->mu, this->Lambda) = CondNormalWishart(Uhat, this->mu0, this->b0, this->WI + lambda_beta * BBt, this->df + beta.cols());
      sample_beta();
      smurff::linop::compute_uhat(Uhat, *Features, beta);
      lambda_beta = sample_lambda_beta(beta, this->Lambda, lambda_beta_nu0, lambda_beta_mu0);
   }

   void addSideInfo(std::shared_ptr<FType> &Fmat, bool comp_FtF = false)
   {
      // side information
      Features = Fmat;
      use_FtF = comp_FtF;

      // initial value (should be determined automatically)
      lambda_beta = 5.0;
      // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
      lambda_beta_mu0 = 1.0;
      lambda_beta_nu0 = 1e-3;
   }

   double getLinkLambda()
   {
      return lambda_beta;
   }

   const Eigen::VectorXd getMu(int n) const override
   {
      return this->mu + Uhat.col(n);
   }

   void compute_Ft_y_omp(Eigen::MatrixXd& Ft_y)
   {
      const int num_feat = beta.cols();

      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + std::sqrt(lambda_beta) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      Eigen::MatrixXd tmp = (U() + MvNormal_prec_omp(Lambda, num_cols())).colwise() - mu;
      Ft_y = smurff::linop::A_mul_B(tmp, *Features);
      Eigen::MatrixXd tmp2 = MvNormal_prec_omp(Lambda, num_feat);

      #pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++)
      {
         for (int d = 0; d < num_latent(); d++)
         {
            Ft_y(d, f) += std::sqrt(lambda_beta) * tmp2(d, f);
         }
      }
   }

   // Update beta and Uhat
   virtual void sample_beta()
   {
      if (use_FtF)
         sample_beta_direct();
      else
         sample_beta_cg();
   }

   void setLambdaBeta(double lb)
   {
      lambda_beta = lb;
   }

   void setTol(double t)
   {
      tol = t;
   }

   void save(std::shared_ptr<const StepFile> sf) const override
   {
      NormalPrior::save(sf);

      std::string path = sf->getPriorFileName(m_mode);
      smurff::matrix_io::eigen::write_matrix(path, this->beta);
   }

   void restore(std::shared_ptr<const StepFile> sf) override
   {
      NormalPrior::restore(sf);

      std::string path = sf->getPriorFileName(m_mode);

      THROWERROR_FILE_NOT_EXIST(path);

      smurff::matrix_io::eigen::read_matrix(path, this->beta);
   }

private:

   std::ostream &printSideInfo(std::ostream &os, const SparseDoubleFeat &F)
   {
      os << "SparseDouble [" << F.rows() << ", " << F.cols() << "]\n";
      return os;
   }

   std::ostream &printSideInfo(std::ostream &os, const Eigen::MatrixXd &F)
   {
      os << "DenseDouble [" << F.rows() << ", " << F.cols() << "]\n";
      return os;
   }

   std::ostream &printSideInfo(std::ostream &os, const SparseFeat &F)
   {
      os << "SparseBinary [" << F.rows() << ", " << F.cols() << "]\n";
      return os;
   }

public:

   std::ostream &info(std::ostream &os, std::string indent) override
   {
      NormalPrior::info(os, indent);
      os << indent << " SideInfo: "; printSideInfo(os, *Features);
      os << indent << " Method: " << (use_FtF ? "Cholesky Decomposition" : "CG Solver") << "\n";
      os << indent << " Tol: " << tol << "\n";
      os << indent << " LambdaBeta: " << lambda_beta << "\n";
      return os;
   }

   std::ostream &status(std::ostream &os, std::string indent) const override
   {
      os << indent << "  " << m_name << ": Beta = " << beta.norm() << "\n";
      return os;
   }

private:

   // direct method
   void sample_beta_direct()
   {
      Eigen::MatrixXd Ft_y;
      this->compute_Ft_y_omp(Ft_y);

      Eigen::MatrixXd K(FtF.rows(), FtF.cols());
      K.triangularView<Eigen::Lower>() = FtF;
      K.diagonal().array() += lambda_beta;
      chol_decomp(K);
      chol_solve_t(K, Ft_y);
      beta = Ft_y;
   }

   // BlockCG solver
   void sample_beta_cg();

public:

   static std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
   {
      const int D = beta.rows();
      Eigen::MatrixXd BB(D, D);
      smurff::linop::A_mul_At_combo(BB, beta);
      double nux = nu + beta.rows() * beta.cols();
      double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace() );
      double b   = nux / 2;
      double c   = 2 * mux / nux;
      return std::make_pair(b, c);
   }

   static double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu)
   {
      auto gamma_post = posterior_lambda_beta(beta, Lambda_u, nu, mu);
      return rgamma(gamma_post.first, gamma_post.second);
   }
};

template<class FType>
void MacauPrior<FType>::sample_beta_cg()
{
   Eigen::MatrixXd Ft_y;
   this->compute_Ft_y_omp(Ft_y);

   smurff::linop::solve_blockcg(beta, *Features, lambda_beta, Ft_y, tol, 32, 8);
}

// specialization for dense matrices --> always direct method
template<>
void MacauPrior<Eigen::Matrix<double, -1, -1, 0, -1, -1>>::sample_beta_cg();

}

//==========================

//macau Probit sample latents
/*
template<class FType>
void MacauPrior<FType>::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                       double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
  const int N = U.cols();
  #pragma omp parallel for schedule(dynamic, 2)
  for(int n = 0; n < N; n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent_blas_probit(U, n, mat, mean_value, samples, mu + Uhat.col(n), Lambda, num_latent);
  }
}
*/

//macau Tensor method
/*
template<class FType>
void MacauPrior<FType>::sample_latents(ProbitNoise& noiseModel, TensorData & data,
                               std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) {
  // TODO:
}
*/
