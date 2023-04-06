#pragma once

#include <Eigen/Core>

namespace NNFSCore
{
    /**
     * @brief Abstract base class for functions with gradients.
     *
     */
    class Differentiable
    {
    public:
        /**
         * @brief Pure virtual function for computing the gradient.
         *
         * @param input_tensor The input tensor.
         * @return MatrixXd The gradient tensor.
         */

        virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd &input_tensor) const = 0;
    };
}