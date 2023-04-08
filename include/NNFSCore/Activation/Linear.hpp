#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "Activation.hpp"

namespace NNFSCore
{
    /**
     * @brief Linear activation function.
     *
     */
    class Linear : public Activation
    {
    public:
        /**
         * @brief Computes the Linear activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The activation tensor.
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) const
        {
            return input_tensor;
        }

        /**
         * @brief Computes the gradient of the Linear activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The gradient tensor.
         */
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &input_tensor) const
        {
            return Eigen::MatrixXd::Ones(input_tensor.rows(), input_tensor.cols());
        }
    };
}

#endif // LINEAR_HPP