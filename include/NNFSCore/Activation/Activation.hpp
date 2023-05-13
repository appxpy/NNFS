#pragma once

#include "../Layer/Layer.hpp"

namespace NNFSCore
{
    /**
     * @brief Enum class for activation types
     */
    enum class ActivationType
    {
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX,
        NONE
    };
    /**
     * @brief Base class for all activation functions
     *
     * @details This class is the base class for all activation functions. It provides the interface for all activation functions.
     */
    class Activation : public Layer
    {
    public:
        ActivationType activation_type; // Type of activation function

    public:
        /**
         * @brief Construct a new Activation object
         *
         * @param activation_type Type of activation function
         */
        Activation(ActivationType activation_type) : Layer(LayerType::ACTIVATION), activation_type(activation_type) {}

    protected:
        Eigen::MatrixXd _forward_input; // Input data for forward pass
    };
} // namespace NNFSCore
