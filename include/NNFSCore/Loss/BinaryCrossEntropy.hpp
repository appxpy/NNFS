#ifndef BINARY_CROSS_ENTROPY_HPP
#define BINARY_CROSS_ENTROPY_HPP

#include "Loss.hpp"

namespace NNFSCore
{

    /**
     * @brief Class for binary cross-entropy loss.
     */
    class BinaryCrossEntropy : public Loss
    {
    public:
        /**
         * @brief Compute the binary cross-entropy loss.
         *
         * @param predictions The input tensor of predictions.
         * @param labels The input tensor of labels.
         * @return Eigen::MatrixXd The computed loss.
         */
        double operator()(const Eigen::MatrixXd &predictions,
                          const Eigen::MatrixXd &labels) const
        {
            double loss = (labels.array() * predictions.array().log() +
                           (1.0 - labels.array()) * (1.0 - predictions.array()).log())
                              .mean();
            return -1.0 *
                   loss;
        }

        /**
         * @brief Compute the gradient of the binary cross-entropy loss.
         *
         * @param predictions The input tensor of predictions.
         * @param labels The input tensor of labels.
         * @return Eigen::MatrixXd The gradient tensor.
         */
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &predictions,
                                 const Eigen::MatrixXd &labels) const
        {
            constexpr double epsilon = 1e-7;
            Eigen::ArrayXXd clipped_predictions = predictions.array().min(1.0 - epsilon).max(epsilon);
            return -1.0 * (labels.array() / clipped_predictions - (1.0 - labels.array()) / (1.0 - clipped_predictions));
        }
    };

}

#endif // BINARY_CROSS_ENTROPY_HPP
