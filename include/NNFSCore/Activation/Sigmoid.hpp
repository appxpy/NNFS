#pragma once

#include "Activation.hpp"

namespace NNFSCore
{
    class Sigmoid : public Activation
    {
    public:
        Sigmoid() : Activation(ActivationType::SIGMOID) {}

        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) override
        {
            _forward_input = x;

            out = 1 / (1 + (-x).array().exp());

            _forward_output = out;
        }

        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) override
        {
            out = _forward_output.array() * (1 - _forward_output.array()) * dx.array();
        }

    private:
        Eigen::MatrixXd _forward_output;
    };
}
