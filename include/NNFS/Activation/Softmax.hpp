#pragma once

#include "Activation.hpp"

namespace NNFS
{
    /**
     * @brief Softmax activation function
     *
     * @details This class implements the softmax activation function.
     */
    class Softmax : public Activation
    {
    public:
        Eigen::MatrixXd _forward_output; // Output data for forward pass

    public:
        /**
         * @brief Construct a new Softmax object
         */
        Softmax() : Activation(ActivationType::SOFTMAX) {}

        /**
         * @brief Forward pass of the softmax activation function
         *
         * @param[out] out Output of the softmax activation function
         * @param[in] x Input to the softmax activation function
         */
        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) override
        {
            _forward_input = x;
            equation(out, x);
            _forward_output = out;
        }

        /**
         * @brief Backward pass of the softmax activation function
         *
         * @param[out] out Input gradient
         * @param[in] dx Output gradient
         */
        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) override
        {
            // Create uninitialized array
            out.resizeLike(dx);

            for (int i = 0; i < _forward_output.rows(); i++)
            {

                Eigen::MatrixXd single_output = _forward_output.row(i).transpose();

                Eigen::MatrixXd jacobian_matrix = Eigen::MatrixXd::Zero(_forward_output.cols(), _forward_output.cols());
                for (int j = 0; j < _forward_output.cols(); ++j)
                {
                    jacobian_matrix(j, j) = single_output(j, 0);
                }
                jacobian_matrix -= single_output * single_output.transpose();

                out.row(i) = jacobian_matrix * dx.row(i).transpose();
            }
        }

        /**
         * @brief Softmax equation
         *
         * @param[out] out Output of the softmax activation function
         * @param[in] x Input to the softmax activation function
         */
        void equation(Eigen::MatrixXd &out, const Eigen::MatrixXd &x)
        {
            Eigen::MatrixXd expX = x;
            for (int i = 0; i < expX.rows(); i++)
            {
                double max_val = expX.row(i).maxCoeff();
                expX.row(i) -= Eigen::VectorXd::Constant(expX.cols(), max_val);
            }

            expX = expX.array().exp();
            out = x;
            for (int row = 0; row < x.rows(); ++row)
            {
                out.row(row) = expX.row(row) / expX.row(row).sum();
            }
        }
    };
} // namespace NNFS
