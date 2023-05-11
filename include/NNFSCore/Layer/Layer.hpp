#ifndef LAYER_BASE_HPP
#define LAYER_BASE_HPP

#include <Eigen/Core>
#include "../Utilities/clue.hpp"

namespace NNFSCore
{
    enum class LayerType
    {
        DENSE,
        ACTIVATION
    };

    /**
     * @class Layer
     * @brief Abstract base class for layers in a neural network
     */
    class Layer
    {
    public:
        LayerType type;

    public:
        Layer(LayerType type) : type(type) {}

        virtual ~Layer() = default;

        virtual void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &input_tensor) = 0;

        virtual void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &grad_output) = 0;
    };
}

#endif