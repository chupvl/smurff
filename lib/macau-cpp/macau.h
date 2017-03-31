#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>

#include "model.h"
#include "latentprior.h"
#include "noisemodels.h"


namespace Macau {

class ILatentPrior;

struct MacauConfig {
    std::string fname_train;
    std::string fname_test;
    std::vector<std::string> fname_row_features;
    std::vector<std::string> fname_col_features;
    std::string row_prior = "default";
    std::string col_prior = "default";
    std::string output_prefix;
    std::string fixed_precision, adaptive_precision;

    int output_freq           = 0; // never
    int burnin                = 200;
    int nsamples              = 800;
    int num_latent            = 96;
    double lambda_beta        = 10.0;
    double tol                = 1e-6;

    double precision          = 5.0;
    double sn_init            = 1.0;
    double sn_max             = 10.0;

    double test_split         = .0;
};

class BaseSession  {
   public:
      std::unique_ptr<INoiseModel>                noise;
      std::vector< std::unique_ptr<ILatentPrior>> priors;
      std::unique_ptr<Factors>                    model;
    
      //-- add model
      template<class Model>
      Model         &addModel(int num_latent);
      SparseMF      &sparseModel(int num_latent);
      SparseBinaryMF&sparseBinaryModel(int num_latent);
      DenseDenseMF  &denseDenseModel(int num_latent);
      SparseDenseMF &sparseDenseModel(int num_latent);

      //-- add priors
      template<class Prior>
      inline Prior& addPrior();

      // set noise models
      FixedGaussianNoise &setPrecision(double p);
      AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);

      void init();
      void step();

      virtual std::ostream &printInitStatus(std::ostream &, std::string indent);

      std::string name;
};

// try adding num_latent as template parameter to Session
class Session : public BaseSession {
  public:
      double      threshold   = NAN;
      bool        verbose     = true;
      int         nsamples    = 100;
      int         burnin      = 50;
      int         save_freq   = 0;
      std::string save_prefix = "model";

      // while running
      int         iter;

      // c'tor
      Session() { name = "MacauSession"; }

      //-- set params
      void setThreshold(double t) { threshold = t; }
      void setSamples(int burnin, int nsamples);
      void setVerbose(bool v) { verbose = v; };
      void setSavePrefix(std::string pref) { save_prefix = pref; };
      void setSaveFrequency(int f) { save_freq = f; };

      void setFromArgs(int argc, char** argv);
      void setFromConfig(MacauConfig &);

      // execution of the sampler
      void init();
      void run();
      void step();

      std::ostream &printInitStatus(std::ostream &, std::string indent) override;

   private:
      void saveModel(int isample);
      void printStatus(double elapsedi);
};

class MPISession : public Session {
  public:
    MPISession() { name = "MPISession"; }
      
    void run();
    std::ostream &printInitStatus(std::ostream &os, std::string indent) override;

    int world_rank;
    int world_size;

};


class PythonSession : public Session {
  public:
    PythonSession() { name = "PythonSession"; }

    void run();

  private:
    static void intHandler(int); 
    static volatile bool keepRunning;

};

template<class Prior>
Prior& BaseSession::addPrior()
{
    auto pos = priors.size();
    Prior *p = new Prior(*this, pos);
    priors.push_back(std::unique_ptr<ILatentPrior>(p));
    return *p;
}

}

#endif /* MACAU_H */
