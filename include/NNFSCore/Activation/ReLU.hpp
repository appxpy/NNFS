#ifndef RELU_ACTIVATION_HPP
#define RELU_ACTIVATION_HPP

#include "Activation.hpp"

namespace NNFSCore
{
    class ReLU : public Activation
    {
    public:
        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) override
        {
            _forward_input = x;

            out = (x.array() < 0.0).select(0.0, x);
        }

        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) override
        {
            out = dx.array() * (_forward_input.array() > 0).cast<double>();
        }
    };
}

#endif