#include <SmurffCpp/Utils/StepFile.h>

#include <iostream>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/result.h>
#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/Utils/Error.h>
#include <SmurffCpp/IO/GenericIO.h>

#define STEP_SAMPLE_PREFIX "-sample-"
#define STEP_CHECKPOINT_PREFIX "-checkpoint-"
#define STEP_INI_SUFFIX "-step.ini"

#define MODEL_PREFIX "model_"
#define PRIOR_PREFIX "prior_"

#define NUM_MODELS_TAG "num_models"
#define NUM_PRIORS_TAG "num_priors"
#define PRED_TAG "pred"
#define PRED_STATE_TAG "pred_state"

using namespace smurff;

StepFile::StepFile(std::int32_t isample, std::string prefix, std::string extension, bool create, bool checkpoint)
   : m_isample(isample), m_prefix(prefix), m_extension(extension), m_checkpoint(checkpoint)
{
   if (create)
   {
      m_iniReader = std::make_shared<INIFile>();
      m_iniReader->create(getStepFileName());
   }
   else
   {
      //load all entries in ini file to be able to go through step file internals
      m_iniReader = std::make_shared<INIFile>();
      m_iniReader->open(getStepFileName());
   }
}

StepFile::StepFile(const std::string& path, std::string prefix, std::string extension)
   : m_prefix(prefix), m_extension(extension)
{
   m_isample = tryGetIsampleFromPathInternal(path, STEP_CHECKPOINT_PREFIX, STEP_INI_SUFFIX);

   if (m_isample < 0)
   {
      m_isample = tryGetIsampleFromPathInternal(path, STEP_SAMPLE_PREFIX, STEP_INI_SUFFIX);

      if (m_isample < 0)
      {
         THROWERROR("Invalid step file name");
      }
      else
      {
         m_checkpoint = false;
      }
   }
   else
   {
      m_checkpoint = true;
   }

   //load all entries in ini file to be able to go through step file internals
   m_iniReader = std::make_shared<INIFile>();
   m_iniReader->open(path);
}

//name methods

std::string StepFile::getStepFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + STEP_INI_SUFFIX;
}

std::int32_t StepFile::tryGetIsampleFromPathInternal(const std::string& path, const std::string& prefix, const std::string& suffix) const
{
   std::size_t idx0 = path.find(prefix);
   if (idx0 == std::string::npos)
      return -1;

   std::size_t idx1 = path.find(suffix);
   THROWERROR_ASSERT_MSG(idx1 != std::string::npos, "Invalid step file name");

   std::size_t start = idx0 + prefix.length();
   std::string indexStr = path.substr(start, idx1 - start);

   std::int32_t index;
   std::stringstream ss;
   ss << indexStr;
   ss >> index;

   return index;
}

std::string StepFile::getStepPrefix() const
{
   std::string prefix = m_checkpoint ? STEP_CHECKPOINT_PREFIX : STEP_SAMPLE_PREFIX;
   return m_prefix + prefix + std::to_string(m_isample);
}

std::string StepFile::getModelFileName(std::uint64_t index) const
{
   std::string prefix = getStepPrefix();
   return prefix + "-U" + std::to_string(index) + "-latents" + m_extension;
}

std::string StepFile::getLinkMatrixFileName(std::uint32_t mode) const
{
   std::string prefix = getStepPrefix();
   return prefix + "-F" + std::to_string(mode) + "-link" + m_extension;
}

std::string StepFile::getPredFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + "-predictions.csv";
}

std::string StepFile::getPredStateFileName() const
{
   std::string prefix = getStepPrefix();
   return prefix + "-predictions-state.ini";
}

//save methods

void StepFile::saveModel(std::shared_ptr<const Model> model) const
{
   model->save(shared_from_this());

   //save models

   appendCommentToStepFile("models");
   appendToStepFile(std::string(), NUM_MODELS_TAG, std::to_string(model->nmodes()));

   for (std::uint64_t mIndex = 0; mIndex < model->nmodes(); mIndex++)
   {
      std::string path = getModelFileName(mIndex);
      appendToStepFile(std::string(), MODEL_PREFIX + std::to_string(mIndex), path);
   }
}

void StepFile::savePred(std::shared_ptr<const Result> m_pred) const
{
   if (m_pred->isEmpty())
      return;

   m_pred->save(shared_from_this());

   //save predictions

   appendCommentToStepFile("predictions");
   appendToStepFile(std::string(), PRED_TAG, getPredFileName());
   appendToStepFile(std::string(), PRED_STATE_TAG, getPredStateFileName());
}

void StepFile::savePriors(const std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   for (auto &p : priors)
   {
      p->save(shared_from_this());
   }

   //save priors

   appendCommentToStepFile("priors");
   appendToStepFile(std::string(), NUM_PRIORS_TAG, std::to_string(priors.size()));

   for (std::uint64_t pIndex = 0; pIndex < priors.size(); pIndex++)
   {
      std::string priorPath = getLinkMatrixFileName(priors.at(pIndex)->getMode());
      appendToStepFile(std::string(), PRIOR_PREFIX + std::to_string(pIndex), priorPath);
   }
}

void StepFile::save(std::shared_ptr<const Model> model, std::shared_ptr<const Result> pred, const std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   saveModel(model);
   savePred(pred);
   savePriors(priors);
}

//restore methods

void StepFile::restoreModel(std::shared_ptr<Model> model) const
{
   //it is enough to check presence of num tag
   auto nmodels = tryGetIniValueBase(NUM_MODELS_TAG);
   if (!nmodels.first)
      return;

   model->restore(shared_from_this());
}

void StepFile::restorePred(std::shared_ptr<Result> m_pred) const
{
   auto predIt = tryGetIniValueBase(PRED_TAG);
   if (!predIt.first)
      return;

   auto predStateIt = tryGetIniValueBase(PRED_STATE_TAG);
   if (!predStateIt.first)
      return;

   m_pred->restore(shared_from_this());
}

void StepFile::restorePriors(std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   //it is enough to check presence of num tag
   auto npriors = tryGetIniValueBase(NUM_PRIORS_TAG);
   if (!npriors.first)
      return;

   for (auto &p : priors)
   {
      p->restore(shared_from_this());
   }
}

void StepFile::restore(std::shared_ptr<Model> model, std::shared_ptr<Result> pred, std::vector<std::shared_ptr<ILatentPrior> >& priors) const
{
   restoreModel(model);
   restorePred(pred);
   restorePriors(priors);
}

//remove methods

void StepFile::removeModel() const
{
   std::uint64_t index = 0;
   while (true)
   {
      std::string path = getModelFileName(index++);
      if (!generic_io::file_exists(path))
         break;

      std::remove(path.c_str());
   }

   std::int32_t nModels = getNSamples();
   for(std::int32_t i = 0; i < nModels; i++)
      removeFromStepFile(MODEL_PREFIX + std::to_string(i));

   removeFromStepFile(NUM_MODELS_TAG);
}

void StepFile::removePred() const
{
   std::remove(getPredFileName().c_str());
   removeFromStepFile(PRED_TAG);

   std::remove(getPredStateFileName().c_str());
   removeFromStepFile(PRED_STATE_TAG);
}

void StepFile::removePriors() const
{
   std::uint32_t mode = 0;
   while (true)
   {
      std::string path = getLinkMatrixFileName(mode++);
      if (!generic_io::file_exists(path))
         break;

      std::remove(path.c_str());
   }

   std::int32_t nPriors = getNPriors();
   for (std::int32_t i = 0; i < nPriors; i++)
      removeFromStepFile(PRIOR_PREFIX + std::to_string(i));

   removeFromStepFile(NUM_PRIORS_TAG);
}

void StepFile::remove(bool model, bool pred, bool priors) const
{
   //remove all model files
   if(model)
      removeModel();

   //remove all pred files
   if(pred)
      removePred();

   //remove all prior files
   if(priors)
      removePriors();

   //remove step file itself
   std::remove(getStepFileName().c_str());

   THROWERROR_ASSERT_MSG(m_iniReader->empty(), "Unexpected data in step file");

   //nullify reader
   m_iniReader = std::shared_ptr<INIFile>();
}

//getters

std::int32_t StepFile::getIsample() const
{
   return m_isample;
}

bool StepFile::getCheckpoint() const
{
   return m_checkpoint;
}

std::int32_t StepFile::getNSamples() const
{
   return std::stoi(getIniValueBase(NUM_MODELS_TAG));
}

std::int32_t StepFile::getNPriors() const
{
   return std::stoi(getIniValueBase(NUM_PRIORS_TAG));
}

//ini methods

std::string StepFile::getIniValueBase(const std::string& tag) const
{
   THROWERROR_ASSERT_MSG(m_iniReader, "Step ini file is not loaded");

   return m_iniReader->get(std::string(), tag);
}

std::pair<bool, std::string> StepFile::tryGetIniValueBase(const std::string& tag) const
{
   THROWERROR_ASSERT_MSG(m_iniReader, "Step ini file is not loaded");

   return m_iniReader->tryGet(std::string(), tag);
}

void StepFile::appendToStepFile(std::string section, std::string tag, std::string value) const
{
   m_iniReader->appendItem(section, tag, value);
   
   flushLast();
}

void StepFile::appendCommentToStepFile(std::string comment) const
{
   m_iniReader->appendComment(comment);
}

void StepFile::removeFromStepFile(std::string tag) const
{
   m_iniReader->removeItem(std::string(), tag);
}

void StepFile::flushLast() const
{
   m_iniReader->flush();
}
