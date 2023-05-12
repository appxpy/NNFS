#pragma once

#include "../Layer/Layer.hpp"

namespace NNFSCore
{

    enum class ActivationType
    {
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX,
        NONE
    };
    /**
     * @brief Abstract base class for activation functions.
     *
     */
    class Activation : public Layer
    {
    public:
        ActivationType activation_type;

    public:
        Activation(ActivationType activation_type) : Layer(LayerType::ACTIVATION), activation_type(activation_type) {}

    protected:
        Eigen::MatrixXd _forward_input;
    };
};
