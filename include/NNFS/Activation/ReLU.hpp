#pragma once

#include "Activation.hpp"

namespace NNFS
{
    /**
     * @brief ReLU activation function
     *
     * @details This class implements the ReLU activation function.
     */
    class ReLU : public Activation
    {
    public:
        /**
         * @brief Construct a new ReLU object
         */
        ReLU() : Activation(ActivationType::RELU) {}

        /**
         * @brief Forward pass of the ReLU activation function
         *
         * @param[out] out Output of the ReLU activation function
         * @param[in] x Input to the ReLU activation function
         */
        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) override
        {
            _forward_input = x;

            out = (x.array() < 0.0).select(0.0, x);
        }

        /**
         * @brief Backward pass of the ReLU activation function
         *
         * @param[out] out Input gradient
         * @param[in] dx Output gradient
         */
        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) override
        {
            out = dx.array() * (_forward_input.array() > 0).cast<double>();
        }
    };
} // namespace NNFS
