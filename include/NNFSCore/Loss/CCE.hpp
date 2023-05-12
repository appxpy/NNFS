#pragma once

#include "Loss.hpp"

namespace NNFSCore
{
    class CCE : public Loss
    {
    public:
        CCE() : Loss(LossType::CCE) {}

        void forward(Eigen::MatrixXd &sample_losses, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            Eigen::MatrixXd clipped_predictions = predictions.array().max(1e-7).min(1 - 1e-7); // clip data to prevent division by zero
            Eigen::MatrixXd correct_confidences = (labels.array() * clipped_predictions.array()).rowwise().sum();
            sample_losses = -(correct_confidences.array().log());
        }

        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            int m = labels.rows(); // number of examples
            out = -labels.array() / predictions.array();
            out /= m;
        }
    };
}

