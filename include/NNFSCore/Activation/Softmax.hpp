#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Activation.hpp"

namespace NNFSCore
{
    /**
     * @brief Softmax activation function.
     *
     */
    class Softmax : public Activation
    {
    public:
        /**
         * @brief Computes the softmax activation of the input tensor.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The activation tensor.
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) const
        {
            Eigen::MatrixXd exp_input = input_tensor.array().exp();
            Eigen::VectorXd col_sum = exp_input.colwise().sum();
            return exp_input.array().colwise() / col_sum.array();
        }

        /**
         * @brief Computes the gradient of the softmax activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The gradient tensor.
         */
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &input_tensor) const
        {
            Eigen::MatrixXd softmax_output = operator()(input_tensor);
            return softmax_output.array() * (Eigen::MatrixXd::Ones(input_tensor.rows(), input_tensor.cols()).array() - softmax_output.array());
        }
    };

#endif // SOFTMAX_H
}