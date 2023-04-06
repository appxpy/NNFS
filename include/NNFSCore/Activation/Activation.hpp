#pragma once

#include "../Common/Differentiable.hpp"

/**
 * @brief Abstract base class for activation functions.
 *
 */

namespace NNFSCore
{
    /**
     * @brief Rectified Linear Unit (ReLU) activation function.
     *
     */
    class Activation : public Differentiable
    {
    public:
        /**
         * @brief Pure virtual function for computing the activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The activation tensor.
         */
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) const = 0;
    };
}
