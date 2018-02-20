#include "ILatentPrior.h"
#include <SmurffCpp/Utils/counters.h>

using namespace smurff;
using namespace Eigen;

ILatentPrior::ILatentPrior(std::shared_ptr<TrainTask> trainTask, uint32_t mode, std::string name)
   : m_trainTask(trainTask), m_mode(mode), m_name(name)
{

}

void ILatentPrior::init()
{
   rrs.init(VectorXd::Zero(num_latent()));
   MMs.init(MatrixXd::Zero(num_latent(), num_latent()));

   //this is some new initialization
   init_Usum();
}

std::shared_ptr<const Model> ILatentPrior::model() const
{
   return m_trainTask->model();
}

std::shared_ptr<Model> ILatentPrior::model()
{
   return m_trainTask->model();
}

double ILatentPrior::predict(const PVec<> &pos) const
{
    return model()->predict(pos);
}

std::shared_ptr<Data> ILatentPrior::data() const
{
   return m_trainTask->data();
}

std::shared_ptr<INoiseModel> ILatentPrior::noise()
{
   return data()->noise();
}

MatrixXd &ILatentPrior::U()
{
   return model()->U(m_mode);
}

const MatrixXd &ILatentPrior::U() const
{
   return model()->U(m_mode);
}

//return V matrices in the model opposite to mode
VMatrixIterator<Eigen::MatrixXd> ILatentPrior::Vbegin()
{
   return model()->Vbegin(m_mode);
}

VMatrixIterator<Eigen::MatrixXd> ILatentPrior::Vend()
{
   return model()->Vend();
}

ConstVMatrixIterator<Eigen::MatrixXd> ILatentPrior::CVbegin() const
{
   return model()->CVbegin(m_mode);
}

ConstVMatrixIterator<Eigen::MatrixXd> ILatentPrior::CVend() const
{
   return model()->CVend();
}

int ILatentPrior::num_latent() const
{
   return model()->nlatent();
}

int ILatentPrior::num_cols() const
{
   return model()->U(m_mode).cols();
}

std::ostream &ILatentPrior::info(std::ostream &os, std::string indent)
{
   os << indent << m_mode << ": " << m_name << "\n";
   return os;
}

bool ILatentPrior::run_slave()
{
   return false;
}

void ILatentPrior::sample_latents()
{
   COUNTER("sample_latents");
   data()->update_pnm(model(), m_mode);

   // for effiency, we keep + update Ucol and UUcol by every thread
   thread_vector<VectorXd> Ucol(VectorXd::Zero(num_latent()));
   thread_vector<MatrixXd> UUcol(MatrixXd::Zero(num_latent(), num_latent()));

   #pragma omp parallel for schedule(guided)
   for(int n = 0; n < U().cols(); n++)
   {
       #pragma omp task
       {
           sample_latent(n);
           const auto& col = U().col(n);
           Ucol.local().noalias() += col;
           UUcol.local().noalias() += col * col.transpose();
       }
   }

   Usum  = Ucol.combine();
   UUsum = UUcol.combine();

   update_prior();
}

void ILatentPrior::save(std::shared_ptr<const StepFile> sf) const
{
}

void ILatentPrior::restore(std::shared_ptr<const StepFile> sf)
{
    init_Usum();
}

void ILatentPrior::init_Usum()
{
    Usum = U().rowwise().sum();
    UUsum = U() * U().transpose(); 
}