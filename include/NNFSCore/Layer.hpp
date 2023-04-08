#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace NNFSCore
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    class Layer
    {
    public:
        virtual ~Layer() = default;

        virtual MatrixXd operator()(const MatrixXd &input_tensor) = 0;
        virtual void build(const MatrixXd &input_tensor) = 0;
        virtual void update(double lr) = 0;
        virtual MatrixXd get_output() const = 0;
    };

    class Dense : public Layer
    {
    public:
        Dense(int units) : _units(units), _input_units(0), _weights(), _bias(), _output(), _dw(), _db() {}

        MatrixXd operator()(const MatrixXd &input_tensor) override
        {
            if (_weights.rows() == 0)
            {
                build(input_tensor);
            }
            _output = _weights * input_tensor + _bias.replicate(1, input_tensor.cols());
            return _output;
        }

        void build(const MatrixXd &input_tensor) override
        {
            _input_units = input_tensor.rows();
            _weights = MatrixXd::Random(_units, _input_units) * std::sqrt(2.0 / _input_units);
            _bias = VectorXd::Zero(_units);
        }

        void update(double lr) override
        {
            _weights -= lr * _dw;
            _bias -= lr * _db;
        }

        MatrixXd get_output() const override
        {
            return _output;
        }

        void set_grad_weights(const MatrixXd &gradients)
        {
            _dw = gradients;
        }

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