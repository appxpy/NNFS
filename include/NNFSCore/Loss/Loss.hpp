#ifndef LOSS_H_
#define LOSS_H_

/**
 * @brief Abstract base class for loss functions.
 */

#include <Eigen/Core>

namespace NNFSCore
{
    class Loss
    {
    public:
        /**
         * @brief Computes the loss given predictions and ground truth labels.
         *
         * @param predictions predicted values (Nx1 Eigen::VectorXd).
         * @param labels ground truth labels (Nx1 Eigen::VectorXd).
         * @return loss value (scalar Eigen::MatrixXd).
         */
        virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &predictions,
                                           const Eigen::VectorXd &labels) const = 0;

        /**
         * @brief Computes the gradient of the loss function w.r.t predictions.
         *
         * @param predictions predicted values (Nx1 Eigen::VectorXd).
         * @param labels ground truth labels (Nx1 Eigen::VectorXd).
         * @return gradient w.r.t predictions (Nx1 Eigen::VectorXd).
         */
        virtual Eigen::VectorXd gradient(const Eigen::VectorXd &predictions,
                                         const Eigen::VectorXd &labels) const = 0;
    };
}

#endif // NN_LOSS_H_
