#pragma once

#include "ScarceMatrixData.h"

namespace smurff
{
   class ScarceBinaryMatrixData : public ScarceMatrixData
   {
   public:
      ScarceBinaryMatrixData(Eigen::SparseMatrix<double>& Y);

   public:
      void get_pnm(const SubModel& model, int mode, int n, Eigen::VectorXd& rr, Eigen::MatrixXd& MM) override;
      void update_pnm(const SubModel& model, int mode) override;

      int nna() const override;
  };
}