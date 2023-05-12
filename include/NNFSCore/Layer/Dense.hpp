#pragma once

#include <iostream>
#include <random>
#include "Layer.hpp"

namespace NNFSCore
{

    /**
     * @class Layer
     * @brief Abstract base class for layers in a neural network
     */
    class Dense : public Layer
    {
    public:
        Dense(int n_input, int n_output,
              double l1_weights_regularizer = .0,
              double l1_biases_regularizer = .0,
              double l2_weights_regularizer = .0,
              double l2_biases_regularizer = .0) : Layer(LayerType::DENSE), _n_input(n_input), _n_output(n_output),
                                                   _l1_weights_regularizer(l1_weights_regularizer),
                                                   _l1_biases_regularizer(l1_biases_regularizer),
                                                   _l2_weights_regularizer(l2_weights_regularizer),
                                                   _l2_biases_regularizer(l2_biases_regularizer)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(-1, 1);

            _weights = Eigen::MatrixXd::Zero(n_input, n_output).unaryExpr([&](double)
                                                                          { return .1 * dis(gen); });
            _biases = Eigen::MatrixXd::Zero(1, n_output);

            _weights_optimizer = Eigen::MatrixXd::Zero(n_input, n_output);

            _biases_optimizer = Eigen::MatrixXd::Zero(1, n_output);

            _weights_optimizer_additional = Eigen::MatrixXd::Zero(n_input, n_output);

            _biases_optimizer_additional = Eigen::MatrixXd::Zero(1, n_output);
        }

        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x)
        {
            _forward_input = x;
            out = x * _weights;
            for (int row = 0; row < out.rows(); ++row)
            {
                out.row(row).array() += _biases.array();
            }
        }

        const Eigen::MatrixXd &dweights() const
        {
            return _dweights;
        }

        const Eigen::MatrixXd &dbiases() const
        {
            return _dbiases;
        }

        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &dx)
        {
            _dweights = _forward_input.transpose() * dx;
            _dbiases = dx.colwise().sum().array();

            // Gradients on regularization
            // L1 on weights
            if (_l1_weights_regularizer > 0)
            {
                Eigen::MatrixXd dL1 = Eigen::MatrixXd::Ones(_weights.rows(), _weights.cols());
                dL1 = (_weights.array() < 0).select(-1, dL1);
                _dweights += _l1_weights_regularizer * dL1;
            }
            // L2 on weights
            if (_l2_weights_regularizer > 0)
            {
                _dweights += 2 * _l2_weights_regularizer * _weights;
            }
            // L1 on biases
            if (_l1_biases_regularizer > 0)
            {
                Eigen::MatrixXd dL1 = Eigen::MatrixXd::Ones(_biases.rows(), _biases.cols());
                dL1 = (_biases.array() < 0).select(-1, dL1);
                _dbiases += _l1_biases_regularizer * dL1.colwise().sum().transpose();
            }
            // L2 on biases
            if (_l2_biases_regularizer > 0)
            {
                _dbiases += 2 * _l2_biases_regularizer * _biases;
            }

            out = dx * _weights.transpose();
        }

        void weights(Eigen::MatrixXd &weights)
        {
            if (_weights.rows() != weights.rows() || _weights.cols() != weights.cols())
            {
                std::cerr << "Shape of new matrix does not match to initial's matrix shape." << std::endl;
            }

            _weights = weights;
        }

        const Eigen::MatrixXd &weights() const
        {
            return _weights;
        }

        void weights_optimizer(Eigen::MatrixXd woptimizer)
        {
            _weights_optimizer = woptimizer;
        }

        void biases_optimizer(Eigen::MatrixXd boptimizer)
        {
            _biases_optimizer = boptimizer;
        }

        const Eigen::MatrixXd &weights_optimizer() const
        {
            return _weights_optimizer;
        }

        const Eigen::MatrixXd &biases_optimizer() const
        {
            return _biases_optimizer;
        }

        void weights_optimizer_additional(Eigen::MatrixXd woptimizer)
        {
            _weights_optimizer_additional = woptimizer;
        }

        void biases_optimizer_additional(Eigen::MatrixXd boptimizer)
        {
            _biases_optimizer_additional = boptimizer;
        }

        const Eigen::MatrixXd &weights_optimizer_additional() const
        {
            return _weights_optimizer_additional;
        }

        const Eigen::MatrixXd &biases_optimizer_additional() const
        {
            return _biases_optimizer_additional;
        }

        const double &l1_weights_regularizer() const
        {
            return _l1_weights_regularizer;
        }

        const double &l2_weights_regularizer() const
        {
            return _l2_weights_regularizer;
        }

        const double &l1_biases_regularizer() const
        {
            return _l1_biases_regularizer;
        }

        const double &l2_biases_regularizer() const
        {
            return _l2_biases_regularizer;
        }

        void biases(Eigen::MatrixXd &biases)
        {
            if (_biases.rows() != biases.rows() || _biases.cols() != biases.cols())
            {
                std::cerr << "Shape of new matrix does not match to initial's matrix shape." << std::endl;
            }
            _biases = biases;
        }

        const Eigen::MatrixXd &biases() const
        {
            return _biases;
        }

        int parameters() const
        {
            return _n_input * _n_output + _n_output;
        }

        void shape(int &n_input, int &n_output) const
        {
            n_input = _n_input;
            n_output = _n_output;
        }

    private:
        int _n_input;
        int _n_output;

        Eigen::MatrixXd _weights;
        Eigen::MatrixXd _biases;

        Eigen::MatrixXd _dweights;
        Eigen::MatrixXd _dbiases;

        Eigen::MatrixXd _weights_optimizer;
        Eigen::MatrixXd _biases_optimizer;

        Eigen::MatrixXd _weights_optimizer_additional;
        Eigen::MatrixXd _biases_optimizer_additional;

        double _l1_weights_regularizer;
        double _l1_biases_regularizer;
        double _l2_weights_regularizer;
        double _l2_biases_regularizer;

        Eigen::MatrixXd _forward_input;
    };
}
