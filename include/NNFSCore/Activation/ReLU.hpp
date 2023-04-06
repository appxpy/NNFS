#pragma once

#include "Activation.hpp"

namespace NNFSCore
{
    /**
     * @brief Rectified Linear Unit (ReLU) activation function.
     *
     */
    class ReLU : public Activation
    {
    public:
        /**
         * @brief Computes the ReLU activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The activation tensor.
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) const
        {
            return input_tensor.cwiseMax(0);
        }

        /**
         * @brief Computes the gradient of the ReLU activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The gradient tensor.
         */
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &input_tensor) const
        {
            Eigen::MatrixXd result = input_tensor;
            result = (input_tensor.array() >= 0).cast<double>();
            return result;
        }
    };
}