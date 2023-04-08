#ifndef BINARY_CROSS_ENTROPY_H_
#define BINARY_CROSS_ENTROPY_H_

#include "Loss.hpp"

namespace NNFSCore
{
    /**
     * @brief Binary cross entropy loss implementation.
     *
     */
    class BinaryCrossEntropy : public Loss
    {
    public:
        /**
         * Compute the loss.
         *
         * @param predictions Predictions from the model.
         * @param labels True labels for the inputs.
         * @return Binary Cross-Entropy loss.
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            return -1 * ((labels.array() * predictions.array().log().array()) + (1 - labels.array() * (1 - predictions.array())).array().log().array()).colwise().mean();
        }
        /**
         * Compute the gradient of the loss with respect to the predictions.
         *
         * @param predictions Predictions from the model.
         * @param labels True labels for the inputs.
         * @return Gradient of the Binary Cross-Entropy loss.
         */
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            return -1 * ((labels.array() / predictions.array()) - ((1 - labels.array()) / (1 - predictions.array())));
        }
    };
}

#endif // BINARY_CROSS_ENTROPY_H_
