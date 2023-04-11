#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Core>

namespace NNFSCore
{

    /**
     * @brief This abstract class must be implemented by concrete Loss classes.
     */
    class Loss
    {
    public:
        /**
         * @brief A pure virtual function for computing the loss.
         *
         * @param predictions The input tensor of predictions.
         * @param labels The input tensor of labels.
         * @return Eigen::MatrixXd The computed loss.
         */
        virtual double operator()(const Eigen::MatrixXd &predictions,
                                  const Eigen::MatrixXd &labels) const = 0;

        /**
         * @brief A pure virtual function for computing the gradient of the loss.
         *
         * @param predictions The input tensor of predictions.
         * @param labels The input tensor of labels.
         * @return Eigen::MatrixXd The gradient tensor.
         */
        virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd &predictions,
                                         const Eigen::MatrixXd &labels) const = 0;
    };

}

#endif // LOSS_HPP
