#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "../Layer/Layer.hpp"

namespace NNFSCore
{
    /**
     * @brief Abstract base class for activation functions.
     *
     */
    class Activation : public Layer
    {
    public:
        Activation() : Layer(LayerType::ACTIVATION) {}

    protected:
        Eigen::MatrixXd _forward_input;
    };
};

#endif // ACTIVATION_HPP