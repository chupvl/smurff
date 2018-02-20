#include "Session.h"

#include <string>

#include <SmurffCpp/Version.h>

#include <SmurffCpp/Utils/omp_util.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/counters.h>
#include <SmurffCpp/Utils/Error.h>

#include <SmurffCpp/DataMatrices/DataCreator.h>
#include <SmurffCpp/Priors/PriorFactory.h>

#include <SmurffCpp/result.h>

using namespace smurff;

ModelReaderTask::ModelReaderTask()
   : m_model(std::make_shared<Model>())
{
}

void ModelReaderTask::setBase(std::shared_ptr<Session> session)
{
   m_config = session->getConfig();
   m_rootFile = session->getRootFile();
}

std::shared_ptr<Model> ModelReaderTask::stepInternal(std::shared_ptr<void> in, bool burnin)
{
   std::shared_ptr<StepFile> stepFile = m_rootFile->openStepFile(stepFileIt->second);
   stepFile->restoreModel(m_model);
   stepFileIt++;
   return m_model;
}

void ModelReaderTask::initBase()
{
   stepFileIt = m_rootFile->stepFilesBegin();

   std::shared_ptr<StepFile> stepFile = m_rootFile->openStepFile(stepFileIt->second);
   std::int32_t nSamples = stepFile->getNSamples();
   std::vector<int> dims(nSamples);
   m_model->init(m_config.getNumLatent(),PVec<>(dims), ModelInitTypes::zero);
}

void ModelReaderTask::printStatusBase(std::string phase, std::int32_t isample, int from)
{

}

void ModelReaderTask::saveBase(std::int32_t isample) const
{

}

void ModelReaderTask::restoreBase()
{
}

std::ostream& ModelReaderTask::infoBase(std::ostream &os, std::string indent) const
{
   return os;
}

const Config& ModelReaderTask::getConfig() const
{
   return m_config;
}

//====

TrainTask::TrainTask()
   : m_model(std::make_shared<Model>())
{
}

void TrainTask::addPrior(std::shared_ptr<ILatentPrior> prior)
{
   prior->setMode(m_priors.size());
   m_priors.push_back(prior);
}

void TrainTask::setBase(std::shared_ptr<Session> session)
{
   m_config = session->getConfig();
   m_rootFile = session->getRootFile();

   std::shared_ptr<ISessionBaseTask> this_task_base = shared_from_this();
   std::shared_ptr<TrainTask> this_task = std::dynamic_pointer_cast<TrainTask>(this_task_base);

   // initialize data

   data_ptr = m_config.getTrain()->create(std::make_shared<DataCreator>(this_task));

   // initialize priors

   std::shared_ptr<IPriorFactory> priorFactory = this->create_prior_factory();
   for (std::size_t i = 0; i < m_config.getPriorTypes().size(); i++)
      this->addPrior(priorFactory->create_prior(this_task, i));
}

std::shared_ptr<Model> TrainTask::stepInternal(std::shared_ptr<void> in, bool burnin)
{
   for (auto &p : m_priors)
      p->sample_latents();
   data()->update(m_model);

   return m_model;
}

void TrainTask::initBase()
{
   //initialize train matrix (centring and noise model)
   data()->init();

   //initialize model (samples)
   m_model->init(m_config.getNumLatent(), data()->dim(), m_config.getModelInitType());

   //initialize priors
   for (auto &p : m_priors)
      p->init();
}

void TrainTask::printStatusBase(std::string phase, std::int32_t isample, int from)
{
   double snorm0 = m_model->U(0).norm();
   double snorm1 = m_model->U(1).norm();

   auto nnz_per_sec = (data()->nnz()) / m_elapsedi;
   auto samples_per_sec = (m_model->nsamples()) / m_elapsedi;

   printf("U:[%1.2e, %1.2e] [took: %0.1fs]\n", snorm0, snorm1, m_elapsedi);

   // avoid computing train_rmse twice
   double train_rmse = NAN;

   if (m_config.getVerbose() > 1)
   {
      train_rmse = data()->train_rmse(m_model);
      printf("  RMSE train: %.4f\n", train_rmse);
      printf("  Priors:\n");

      for (const auto &p : m_priors)
         p->status(std::cout, "     ");

      printf("  Model:\n");
      m_model->status(std::cout, "    ");
      printf("  Noise:\n");
      data()->status(std::cout, "    ");
   }

   if (m_config.getVerbose() > 2)
   {
      printf("  Compute Performance: %.0f samples/sec, %.0f nnz/sec\n", samples_per_sec, nnz_per_sec);
   }

   if (m_config.getCsvStatus().size())
   {
      // train_rmse is printed as NAN, unless verbose > 1
      auto f = fopen(m_config.getCsvStatus().c_str(), "a");
      fprintf(f, "%s;%d;%d;%.4f;%1.2e;%1.2e;%0.1f\n",
         phase.c_str(), isample, from, train_rmse, snorm0, snorm1, m_elapsedi);
      fclose(f);
   }
}

void TrainTask::saveBase(std::int32_t isample) const
{
   std::shared_ptr<StepFile> stepFile = m_rootFile->createStepFile(isample);

   if (m_config.getVerbose())
      printf("-- Saving models into '%s'.\n", stepFile->getStepFileName().c_str());

   stepFile->saveModel(m_model);
   stepFile->savePriors(m_priors);
}

void TrainTask::restoreBase()
{
   std::shared_ptr<StepFile> stepFile = m_rootFile->openLastStepFile();
   if (stepFile)
   {
      if (m_config.getVerbose())
         printf("-- Restoring model from '%s'.\n", stepFile->getStepFileName().c_str());

      stepFile->restoreModel(m_model);
      stepFile->restorePriors(m_priors);
   }
}

std::ostream& TrainTask::infoBase(std::ostream &os, std::string indent) const
{
   os << indent << "  Data: {\n";
   data()->info(os, indent + "    ");
   os << indent << "  }\n";

   os << indent << "  Model: {\n";
   m_model->info(os, indent + "    ");
   os << indent << "  }\n";

   os << indent << "  Priors: {\n";
   for (auto &p : m_priors)
      p->info(os, indent + "    ");
   os << indent << "  }\n";

   return os;
}


MatrixConfig TrainTask::getSample(int mode)
{
   THROWERROR_NOTIMPL_MSG("getSample is unimplemented");
}

std::shared_ptr<Data> TrainTask::data() const
{
   THROWERROR_ASSERT(data_ptr != 0);

   return data_ptr;
}

std::shared_ptr<const Model> TrainTask::model() const
{
   return m_model;
}

std::shared_ptr<Model> TrainTask::model()
{
   return m_model;
}

std::shared_ptr<IPriorFactory> TrainTask::create_prior_factory() const
{
   return std::make_shared<PriorFactory>();
}

const Config& TrainTask::getConfig() const
{
   return m_config;
}

//===

PredictTask::PredictTask()
   : m_pred(std::make_shared<Result>())
{
}

void PredictTask::setBase(std::shared_ptr<Session> session)
{
   m_config = session->getConfig();
   m_rootFile = session->getRootFile();

   // initialize pred

   if (m_config.getClassify())
      m_pred->setThreshold(m_config.getThreshold());

   if (m_config.getTest())
      m_pred->set(m_config.getTest());
}

void PredictTask::initBase()
{

}

std::shared_ptr<Result> PredictTask::stepInternal(std::shared_ptr<Model> in, bool burnin)
{
   //WARNING: update is an expensive operation because of sort (when calculating AUC)
   m_pred->update(in, burnin);
   return m_pred;
}

void PredictTask::printStatusBase(std::string phase, std::int32_t isample, int from)
{
   printf("RMSE: %.4f (1samp: %.4f)", m_pred->rmse_avg, m_pred->rmse_1sample);

   if (m_config.getClassify())
      printf(" AUC:%.4f (1samp: %.4f)", m_pred->auc_avg, m_pred->auc_1sample);

   printf(" [took: %0.1fs]\n", m_elapsedi);

   if (m_config.getCsvStatus().size())
   {
      // train_rmse is printed as NAN, unless verbose > 1
      auto f = fopen(m_config.getCsvStatus().c_str(), "a");
      fprintf(f, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;%0.1f\n",
         phase.c_str(), isample, from, m_pred->rmse_avg, m_pred->rmse_1sample,m_pred->auc_1sample, m_pred->auc_avg, m_elapsedi);
      fclose(f);
   }
}

void PredictTask::saveBase(std::int32_t isample) const
{
   std::shared_ptr<StepFile> stepFile = m_rootFile->openStepFile(isample);

   if (m_config.getVerbose())
      printf("-- Saving predictions into '%s'.\n", stepFile->getStepFileName().c_str());

   stepFile->savePred(m_pred);
}

void PredictTask::restoreBase()
{
   //TODO:
   //last step file should be restored only if we are continuing training with TrainTask
   //it should not be restored if we are using ModelReaderTask

   std::shared_ptr<StepFile> stepFile = m_rootFile->openLastStepFile();
   if (stepFile)
   {
      if (m_config.getVerbose())
         printf("-- Restoring predictions from '%s'.\n", stepFile->getStepFileName().c_str());

      stepFile->restorePred(m_pred);
   }
}

std::ostream& PredictTask::infoBase(std::ostream &os, std::string indent) const
{
   os << indent << "  Result: {\n";
   m_pred->info(os, indent + "    ");
   os << indent << "  }\n";

   return os;
}

std::shared_ptr<std::vector<ResultItem> > PredictTask::getResult()
{
   return m_pred->m_predictions;
}

double PredictTask::getRmseAvg()
{
   return m_pred->rmse_avg;
}

const Config& PredictTask::getConfig() const
{
   return m_config;
}

//===

void Session::setFromRootPath(std::string rootPath)
{
   // assign config

   m_rootFile = std::make_shared<RootFile>(rootPath);
   m_rootFile->restoreConfig(m_config);

   m_config.validate();

   // initialize each task
   for (auto task : m_pipeline)
   {
      task->setBase(shared_from_this());
   }
}

void Session::setFromConfig(const Config& cfg)
{
   // assign config

   cfg.validate();
   m_config = cfg;

   m_rootFile = std::make_shared<RootFile>(m_config.getSavePrefix(), m_config.getSaveExtension());
   m_rootFile->saveConfig(m_config);

   // initialize each task
   for (auto task : m_pipeline)
   {
      task->setBase(shared_from_this());
   }   
}

void Session::init()
{
   m_iter = 0;

   //init omp
   threads_init();

   //init random generator
   if(m_config.getRandomSeedSet())
      init_bmrng(m_config.getRandomSeed());
   else
      init_bmrng();

   // initialize each task
   for (auto task : m_pipeline)
      task->initBase();

   //write header to status file
   if (m_config.getCsvStatus().size())
   {
      auto f = fopen(m_config.getCsvStatus().c_str(), "w");
      fprintf(f, "phase;iter;phase_len;globmean_rmse;colmean_rmse;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;U0;U1;elapsed\n");
      fclose(f);
   }

   //write info to console
   if (m_config.getVerbose())
      info(std::cout, "");

   //restore session (model, priors)
   restore();

   //print session status to console
   if (m_config.getVerbose())
   {
      printStatus();

      printf(" ====== Sampling (burning phase) ====== \n");
   }

   is_init = true;
}

void Session::run()
{
   init();

   while (m_iter < m_config.getBurnin() + m_config.getNSamples())
      step();
}

void Session::step()
{
   THROWERROR_ASSERT(is_init);

   if (m_config.getVerbose() && m_iter == m_config.getBurnin())
   {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
   }

   //evaluate each task
   std::shared_ptr<void> in = nullptr;
   for (auto task : m_pipeline)
   {
      in = task->stepBase(in, m_iter < m_config.getBurnin());
   }

   printStatus();

   save(m_iter - m_config.getBurnin() + 1);

   m_iter++;
}

void Session::printStatus() const
{
   if (!m_config.getVerbose())
      return;

   std::string phase;
   std::int32_t isample, from;
   if (m_iter < 0)
   {
      phase = "Initial";
      isample = 0;
      from = 0;
   }
   else if (m_iter < m_config.getBurnin())
   {
      phase = "Burnin";
      isample = m_iter + 1;
      from = m_config.getBurnin();
   }
   else
   {
      phase = "Sample";
      isample = m_iter - m_config.getBurnin() + 1;
      from = m_config.getNSamples();
   }

   printf("%s %3d/%3d:\n", phase.c_str(), isample, from);

   // print status of each task
   for (auto task : m_pipeline)
      task->printStatusBase(phase, isample, from);
}

std::ostream& Session::info(std::ostream &os, std::string indent) const
{
   os << indent << name << " {\n";

   // print information about each task
   for (auto task : m_pipeline)
      task->infoBase(os, indent);

   os << indent << "  Version: " << smurff::SMURFF_VERSION << "\n" ;
   os << indent << "  Iterations: " << m_config.getBurnin() << " burnin + " << m_config.getNSamples() << " samples\n";

   if (m_config.getSaveFreq() != 0)
   {
      if (m_config.getSaveFreq() > 0) 
      {
          os << indent << "  Save model: every " << m_config.getSaveFreq() << " iteration\n";
      } 
      else 
      {
          os << indent << "  Save model after last iteration\n";
      }
      os << indent << "  Save prefix: " << m_config.getSavePrefix() << "\n";
      os << indent << "  Save extension: " << m_config.getSaveExtension() << "\n";
   }
   else
   {
      os << indent << "  Save model: never\n";
   }

   os << indent << "}\n";
   return os;
}

void Session::save(std::int32_t isample) const
{
   if (!m_config.getSaveFreq() || isample < 0) //do not save if (never save) mode is selected or if burnin
      return;

   //save_freq > 0: check modulo
   if (m_config.getSaveFreq() > 0 && ((isample + 1) % m_config.getSaveFreq()) != 0) //do not save if not a save iteration
      return;

   //save_freq < 0: save last iter
   if (m_config.getSaveFreq() < 0 && isample < m_config.getNSamples()) //do not save if (final model) mode is selected and not a final iteration
      return;

   // save each task
   for (auto task : m_pipeline)
      task->saveBase(isample);
}

void Session::restore()
{
   // restore each task
   for (auto task : m_pipeline)
      task->restoreBase();

   //TODO: 
   //m_iter should be controlled by TrainTask
   //same way as stepFileIt is controlled by ModelReaderTask
   //last iteration should be restored only if we are continuing training with TrainTask

   //restore last iteration index
   std::shared_ptr<StepFile> stepFile = m_rootFile->openLastStepFile();
   if (stepFile)
   {
      m_iter = stepFile->getIsample() + m_config.getBurnin() - 1; //restore original state
      m_iter++; //go to next iteration
   }
}