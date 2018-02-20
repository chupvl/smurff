#pragma once

#include <iostream>
#include <memory>

#include "BaseSession.h"
#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Priors/IPriorFactory.h>
#include <SmurffCpp/Utils/RootFile.h>
#include <SmurffCpp/Utils/counters.h>

namespace smurff {

class SessionFactory;

class Session;
struct Config;

class ISessionBaseTask : public std::enable_shared_from_this<ISessionBaseTask>
{
public:
   virtual ~ISessionBaseTask()
   {
   }

public:
   virtual std::shared_ptr<void> stepBase(std::shared_ptr<void> in, bool burnin) = 0;

   virtual void initBase() = 0;

   virtual void setBase(std::shared_ptr<Session> session) = 0;

   virtual void saveBase(std::int32_t isample) const = 0;

   virtual void restoreBase() = 0;

   virtual void printStatusBase(std::string phase, std::int32_t isample, int from) = 0;

   virtual std::ostream &infoBase(std::ostream &, std::string indent) const = 0;
};

template<typename Tin, typename Tout>
class ISessionTask : public ISessionBaseTask
{
protected:
   double m_elapsedi = 0;

public:
   virtual ~ISessionTask()
   {
   }

protected:
   virtual std::shared_ptr<Tout> stepInternal(std::shared_ptr<Tin> in, bool burnin) = 0;

protected:
   std::shared_ptr<void> stepBase(std::shared_ptr<void> in, bool burnin) override
   {
      double starti = tick();
      std::shared_ptr<void> res = std::static_pointer_cast<void>(stepInternal(std::static_pointer_cast<Tin>(in), burnin));
      double endi = tick();

      m_elapsedi = endi - starti;

      return res;
   }
};

class ModelReaderTask : public ISessionTask<void, Model>
{
protected:
   std::shared_ptr<Model> m_model;

   std::shared_ptr<const RootFile> m_rootFile;

   Config m_config;

   std::vector<std::pair<std::string, std::string> >::const_iterator stepFileIt;

public:
   ModelReaderTask();

protected:
   void setBase(std::shared_ptr<Session> session) override;

   std::shared_ptr<Model> stepInternal(std::shared_ptr<void> in, bool burnin) override;

   void initBase() override;

   void printStatusBase(std::string phase, std::int32_t isample, int from) override;

   void saveBase(std::int32_t isample) const override;

   void restoreBase() override;

   std::ostream& infoBase(std::ostream &os, std::string indent) const override;

public:
   const Config& getConfig() const;
};

class TrainTask : public ISessionTask<void, Model>
{
protected:
   std::shared_ptr<Model> m_model;

   std::shared_ptr<const RootFile> m_rootFile;

   std::vector<std::shared_ptr<ILatentPrior> > m_priors;

   std::shared_ptr<Data> data_ptr;

   Config m_config;

public:
   TrainTask();

private:
   void addPrior(std::shared_ptr<ILatentPrior> prior);

protected:
   void setBase(std::shared_ptr<Session> session) override;

   std::shared_ptr<Model> stepInternal(std::shared_ptr<void> in, bool burnin) override;

   void initBase() override;

   void printStatusBase(std::string phase, std::int32_t isample, int from) override;

   void saveBase(std::int32_t isample) const override;

   void restoreBase() override;

   std::ostream& infoBase(std::ostream &os, std::string indent) const override;

public:
   MatrixConfig getSample(int mode);

public:
   std::shared_ptr<Data> data() const;

   std::shared_ptr<const Model> model() const;

   std::shared_ptr<Model> model();

public:
   virtual std::shared_ptr<IPriorFactory> create_prior_factory() const;

public:
   const Config& getConfig() const;
};

class PredictTask : public ISessionTask<Model, Result>
{
protected:
   std::shared_ptr<Result> m_pred;

   std::shared_ptr<const RootFile> m_rootFile;

   Config m_config;

public:
   PredictTask();

protected:
   void setBase(std::shared_ptr<Session> session) override;

   void initBase() override;

   std::shared_ptr<Result> stepInternal(std::shared_ptr<Model> in, bool burnin) override;

   void printStatusBase(std::string phase, std::int32_t isample, int from) override;

   void saveBase(std::int32_t isample) const;

   void restoreBase() override;

   std::ostream& infoBase(std::ostream &os, std::string indent) const override;

public:
   std::shared_ptr<std::vector<ResultItem> > getResult();

   double getRmseAvg();

public:
   const Config& getConfig() const;
};

class Session : public ISession, public std::enable_shared_from_this<Session>
{
   //only session factory should call setFromConfig
   friend class SessionFactory;

protected:
   std::string name;

   std::vector<std::shared_ptr<ISessionBaseTask>> m_pipeline;

protected:
   bool is_init = false;

private:
   std::shared_ptr<RootFile> m_rootFile;

private:
   Config m_config;

   int m_iter = -1; //index of step iteration

protected:
   Session()
   {
      name = "Session";
   }

   virtual ~Session()
   {
   }

   //creation from config
protected:
   void setFromRootPath(std::string rootPath);

   void setFromConfig(const Config& cfg);

   // execution of the sampler
public:
   void run() override;

protected:
   void init() override;

   void step();

protected:
   void printStatus() const;

protected:
   //save iteration
   void save(std::int32_t isample) const;

   void restore();

public:
   std::ostream& Session::info(std::ostream &os, std::string indent) const;

public:
   void addTask(std::shared_ptr<ISessionBaseTask> task)
   {
      m_pipeline.push_back(task);
   }

public:
   const Config& getConfig()
   {
      return m_config;
   }

   std::shared_ptr<const RootFile> getRootFile() const
   {
      return m_rootFile;
   }
};

}