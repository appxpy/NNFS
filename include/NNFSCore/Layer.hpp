#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace NNFSCore
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    /**
     * @class Layer
     * @brief An abstract base class representing a neural network layer.
     */
    class Layer
    {
    public:
        /**
         * @brief Destructor.
         */
        virtual ~Layer() = default;

        /**
         * @brief Pure virtual function for forward propagation.
         * @param input_tensor The input tensor (matrix) for the layer.
         * @return The output tensor (matrix) after forward propagation.
         */
        virtual MatrixXd operator()(const MatrixXd &input_tensor) = 0;

        /**
         * @brief Pure virtual function to build the layer.
         * @param input_tensor The input tensor (matrix) for the layer.
         */
        virtual void build(const MatrixXd &input_tensor) = 0;

        /**
         * @brief Pure virtual function to update the layer's parameters.
         * @param lr The learning rate.
         */
        virtual void update(double lr) = 0;

        /**
         * @brief Pure virtual function to get the output tensor of the layer.
         * @return The output tensor (matrix) of the layer.
         */
        virtual MatrixXd get_output() const = 0;
    };

    /**
     * @class Dense
     * @brief A dense (fully connected) layer derived from the Layer base class.
     */
    class Dense : public Layer
    {
    public:
        /**
         * @brief Constructor.
         * @param units The number of output units in the dense layer.
         */
        Dense(int units) : _units(units), _input_units(0), _weights(), _bias(), _output(), _dw(), _db() {}

        /**
         * @brief Forward propagation for the dense layer.
         * @param input_tensor The input tensor (matrix) for the layer.
         * @return The output tensor (matrix) after forward propagation.
         */
        MatrixXd operator()(const MatrixXd &input_tensor) override
        {
            if (_weights.rows() == 0)
            {
                build(input_tensor);
            }
            _output = _weights * input_tensor + _bias.replicate(1, input_tensor.cols());
            return _output;
        }

        /**
         * @brief Builds the dense layer.
         * @param input_tensor The input tensor (matrix) for the layer.
         */
        void build(const MatrixXd &input_tensor) override
        {
            _input_units = input_tensor.rows();
            _weights = MatrixXd::Random(_units, _input_units) * std::sqrt(2.0 / _input_units);
            _bias = VectorXd::Zero(_units);
        }

        /**
         * @brief Updates the layer's parameters.
         * @param lr The learning rate.
         */
        void update(double lr) override
        {
            _weights -= lr * _dw;
            _bias -= lr * _db;
        }

        /**
         * @brief Gets the output tensor of the layer.
         * @return The output tensor (matrix) of the layer.
         */
        MatrixXd get_output() const override
        {
            return _output;
        }

        /**
         * @brief Sets the gradient for the weights.
         * @param gradients The gradient tensor (matrix) for the weights.
         */
        void set_grad_weights(const MatrixXd &gradients)
        {
            _dw = gradients;
        }

        /**
         * @brief Sets the gradient for the bias.
         * @param gradients The gradient tensor (matrix) for the bias.
         */
        void set_grad_bias(const MatrixXd &gradients)
        {
            _db = gradients;
        }

    private:
        int _units;
        int _input_units;
        MatrixXd _weights;
        VectorXd _bias;
        MatrixXd _output;
        MatrixXd _dw;
        MatrixXd _db;
    };
}

#endif // LAYER_HPP