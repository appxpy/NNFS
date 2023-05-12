#pragma once

#include "Activation.hpp"

namespace NNFSCore
{
    class Tanh : public Activation
    {
    public:
        Tanh() : Activation(ActivationType::TANH) {}

        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) override
        {
            _forward_input = x;

            out = x.array().tanh();

            _forward_output = out;
        }

        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) override
        {
            out = (1.0 - _forward_output.array().square()) * dx.array();
        }

    private:
        Eigen::MatrixXd _forward_output;
    };
}
