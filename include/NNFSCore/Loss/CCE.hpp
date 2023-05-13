#pragma once

#include "Loss.hpp"

namespace NNFSCore
{
    /**
     * @brief Cross-entropy loss function
     *
     * @details This class implements the cross-entropy loss function.
     */
    class CCE : public Loss
    {
    public:
        /**
         * @brief Construct a new CCE object
         */
        CCE() : Loss(LossType::CCE) {}

        /**
         * @brief Forward pass of the CCE loss function
         *
         * @param[out] sample_losses Sample losses
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        void forward(Eigen::MatrixXd &sample_losses, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            Eigen::MatrixXd clipped_predictions = predictions.array().max(1e-7).min(1 - 1e-7); // clip data to prevent division by zero
            Eigen::MatrixXd correct_confidences = (labels.array() * clipped_predictions.array()).rowwise().sum();
            sample_losses = -(correct_confidences.array().log());
        }

        /**
         * @brief Backward pass of the CCE loss function
         *
         * @param[out] out Output gradient
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            int m = labels.rows();
            out = -labels.array() / predictions.array();
            out /= m;
        }
    };
} // namespace NNFSCore
