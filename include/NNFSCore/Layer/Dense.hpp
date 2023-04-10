#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"
#include <Eigen/Dense>

namespace NNFSCore
{

    /**
     * @class Dense
     * @brief Fully connected layer
     * @details Inherits from the Layer base class
     */
    class Dense : public Layer
    {
    public:
        /**
         * @brief Constructor for Dense layer
         * @param units Number of units in the layer
         */
        explicit Dense(int units) : _units(units), _input_units(0) {}

        /**
         * @brief Get the output tensor
         * @return The output tensor
         */
        const Eigen::MatrixXd &output() const override { return _output; }

        /**
         * @brief Compute the output of the layer given an input tensor
         * @param input_tensor Input tensor
         * @return The output tensor
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) override
        {
            if (_weights.rows() == 0)
            {
                build(input_tensor);
            }

            _output = _weights * input_tensor + _bias.replicate(1, input_tensor.cols());

            return _output;
        }

        /**
         * @brief Build the layer using the input tensor's shape
         * @param input_tensor Input tensor
         */
        void build(const Eigen::MatrixXd &input_tensor) override
        {
            _input_units = input_tensor.rows();

            // Initialize weights randomly
            _weights = Eigen::MatrixXd::Random(_units, _input_units);
            _weights *= std::sqrt(2.0 / _input_units);

            // Initialize biases to zero
            _bias = Eigen::MatrixXd::Zero(_units, 1);

            // Initialize gradients to zero
            _dw = Eigen::MatrixXd::Zero(_units, _input_units);
            _db = Eigen::MatrixXd::Zero(_units, 1);
        }

        /**
         * @brief Update the layer's parameters using the learning rate
         * @param lr Learning rate
         */
        void update(double lr) override
        {
            _weights -= lr * _dw;
            _bias -= lr * _db;
        }

        /**
         * @brief Get the gradient of the weights
         * @return Gradient of the weights
         */
        const Eigen::MatrixXd &grad_weights() const override
        {
            return _dw;
        }

        /**
         * @brief Set the gradient of the weights
         * @param gradients Gradient of the weights
         */
        void grad_weights(const Eigen::MatrixXd &gradients) override
        {
            _dw = gradients;
        }

        /**
         * @brief Get the gradient of the bias
         * @return Gradient of the bias
         */
        const Eigen::MatrixXd &grad_bias() const override
        {
            return _db;
        }

        /**
         * @brief Set the gradient of the bias
         * @param gradients Gradient of the bias
         */
        void grad_bias(const Eigen::MatrixXd &gradients) override
        {
            _db = gradients;
        }

        /**
         * @brief Get the weights tensor
         * @return Weights tensor
         */
        const Eigen::MatrixXd &weights() const override
        {
            return _weights;
        }

        /**
         * @brief Get the bias tensor
         * @return Bias tensor
         */
        const Eigen::MatrixXd &bias() const override
        {
            return _bias;
        }

    private:
        int _units;
        int _input_units;
        Eigen::MatrixXd _weights;
        Eigen::MatrixXd _bias;
        Eigen::MatrixXd _output;
        Eigen::MatrixXd _dw;
        Eigen::MatrixXd _db;
    };

} // namespace NNFSCore

#endif // DENSE_HPP