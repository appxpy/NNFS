#pragma once

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"

namespace NNFSCore
{
    /**
     * @brief Enum class for layer types
     */
    enum class LayerType
    {
        DENSE,
        ACTIVATION
    };

    /**
     * @brief Base class for all layers
     *
     * @details This class is the base class for all layers. It provides the interface for all layers.
     */
    class Layer
    {
    public:
        LayerType type; // Type of layer

    public:
        /**
         * @brief Construct a new Layer object
         *
         * @param type Type of layer
         */
        Layer(LayerType type) : type(type) {}

        /**
         * @brief Basic destructor
         */
        virtual ~Layer() = default;

        /**
         * @brief Forward pass of the layer
         *
         * @param[out] out Output data
         * @param[in] x Input data
         */
        virtual void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x) = 0;

        /**
         * @brief Backward pass of the layer
         *
         * @param[out] out Input gradient
         * @param[in] dx Output gradient
         */
        virtual void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx) = 0;
    };
} // namespace NNFSCore
