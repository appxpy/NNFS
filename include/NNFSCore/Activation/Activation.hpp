#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "../Common/Differentiable.hpp"

namespace NNFSCore
{
    /**
     * @brief Abstract base class for activation functions.
     *
     */
    class Activation : public Differentiable
    {
    public:
        /**
         * @brief Pure virtual function for computing the activation.
         *
         * @param input_tensor The input tensor.
         * @return Eigen::MatrixXd The activation tensor.
         */
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) const = 0;
    };
}

#endif // ACTIVATION_HPP