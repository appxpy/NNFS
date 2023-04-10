#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <memory>

namespace NNFSCore
{

    /**
     * @class Layer
     * @brief Abstract base class for layers in a neural network
     */
    class Layer
    {
    public:
        virtual ~Layer() = default;

        /**
         * @brief Get the output tensor
         * @return The output tensor
         */
        virtual const Eigen::MatrixXd &output() const = 0;

        /**
         * @brief Compute the output of the layer given an input tensor
         * @param input_tensor Input tensor
         * @return The output tensor
         */
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) = 0;

        /**
         * @brief Build the layer using the input tensor's shape
         * @param input_tensor Input tensor
         */
        virtual void build(const Eigen::MatrixXd &input_tensor) = 0;

        /**
         * @brief Update the layer's parameters using the learning rate
         * @param lr Learning rate
         */
        virtual void update(double lr) = 0;

        /**
         * @brief Get the gradient of the weights
         * @return Gradient of the weights
         */
        virtual const Eigen::MatrixXd &grad_weights() const = 0;

        /**
         * @brief Set the gradient of the weights
         * @param gradients Gradient of the weights
         */
        virtual void grad_weights(const Eigen::MatrixXd &gradients) = 0;

        /**
         * @brief Get the gradient of the bias
         * @return Gradient of the bias
         */
        virtual const Eigen::MatrixXd &grad_bias() const = 0;

        /**
         * @brief Set the gradient of the bias
         * @param gradients Gradient of the bias
         */
        virtual void grad_bias(const Eigen::MatrixXd &gradients) = 0;

        /**
         * @brief Get the weights tensor
         * @return Weights tensor
         */
        virtual const Eigen::MatrixXd &weights() const = 0;

        /**
         * @brief Get the bias tensor
         * @return Bias tensor
         */
        virtual const Eigen::MatrixXd &bias() const = 0;
    };

}

#endif // LAYER_HPP