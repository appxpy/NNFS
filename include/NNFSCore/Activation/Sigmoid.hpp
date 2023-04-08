#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "Activation.hpp"

namespace NNFSCore
{
    /**
     * @brief Sigmoid activation function.
     *
     */
    class Sigmoid : public Activation
    {
    public:
        /**
         * @brief Computes the Sigmoid activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The activation tensor.
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) const
        {
            return 1.0 / (1.0 + (-1.0 * input_tensor.array()).exp());
        }

        /**
         * @brief Computes the gradient of the Sigmoid activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The gradient tensor.
         */
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &input_tensor) const
        {
            Eigen::MatrixXd result = (*this)(input_tensor);
            return result.array() * (1.0 - result.array());
        }
    };
}

#endif // SIGMOID_HPP