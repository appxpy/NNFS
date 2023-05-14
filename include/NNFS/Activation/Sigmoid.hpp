#pragma once

#include "Activation.hpp"

namespace NNFS
{
    /**
     * @brief Sigmoid activation function
     *
     * @details This class implements the sigmoid activation function.
     */
    class Sigmoid : public Activation
    {
    public:
        /**
         * @brief Construct a new Sigmoid object
         */
        Sigmoid() : Activation(ActivationType::SIGMOID) {}

        /**
         * @brief Forward pass of the sigmoid activation function
         *
         * @param[out] out Output of the sigmoid activation function
         * @param[in] x Input to the sigmoid activation function
         */
        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) override
        {
            _forward_input = x;

            out = 1 / (1 + (-x).array().exp());

            _forward_output = out;
        }

        /**
         * @brief Backward pass of the sigmoid activation function
         *
         * @param[out] out Input gradient
         * @param[in] dx Output gradient
         */
        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) override
        {
            out = _forward_output.array() * (1 - _forward_output.array()) * dx.array();
        }

    private:
        Eigen::MatrixXd _forward_output; // Output data for forward pass
    };
} // namespace NNFS
