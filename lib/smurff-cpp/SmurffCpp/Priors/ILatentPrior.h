#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <SmurffCpp/Sessions/Session.h>
#include <SmurffCpp/Noises/INoiseModel.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/ThreadVector.hpp>

#include <SmurffCpp/Model.h>

#include <SmurffCpp/VMatrixIterator.hpp>
#include <SmurffCpp/ConstVMatrixIterator.hpp>

namespace smurff {

class StepFile;

class ILatentPrior
{
public:
   std::shared_ptr<TrainTask> m_trainTask;
   std::uint32_t m_mode;
   std::string m_name = "xxxx";

   smurff::thread_vector<Eigen::VectorXd> rrs;
   smurff::thread_vector<Eigen::MatrixXd> MMs;

protected:
   ILatentPrior(){}

public:
   ILatentPrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode, std::string name = "xxxx");
   virtual ~ILatentPrior() {}
   virtual void init();

   // utility
   std::shared_ptr<const Model> model() const;
   std::shared_ptr<Model> model();

   std::shared_ptr<Data> data() const;
   double predict(const PVec<> &) const;

   std::shared_ptr<INoiseModel> noise();

   Eigen::MatrixXd &U();
   const Eigen::MatrixXd &U() const;

   //return V matrices in the model opposite to mode
   VMatrixIterator<Eigen::MatrixXd> Vbegin();
   
   VMatrixIterator<Eigen::MatrixXd> Vend();

   ConstVMatrixIterator<Eigen::MatrixXd> CVbegin() const;
   
   ConstVMatrixIterator<Eigen::MatrixXd> CVend() const;

   int num_latent() const;
   int num_cols() const;

   const Eigen::VectorXd& getUsum() { return Usum; } 
   const Eigen::MatrixXd& getUUsum()  { return UUsum; }

   virtual void save(std::shared_ptr<const StepFile> sf) const;
   virtual void restore(std::shared_ptr<const StepFile> sf);
   virtual std::ostream &info(std::ostream &os, std::string indent);
   virtual std::ostream &status(std::ostream &os, std::string indent) const = 0;

   // work
   virtual bool run_slave(); // returns true if some work happened...

   virtual void sample_latents();
   virtual void sample_latent(int n) = 0;

   virtual void update_prior() = 0;

private:
   void init_Usum();
   Eigen::VectorXd Usum;
   Eigen::MatrixXd UUsum;

public:
   void setMode(std::uint32_t value)
   {
      m_mode = value;
   }

public:
   std::uint32_t getMode() const
   {
      return m_mode;
   }
};
}
