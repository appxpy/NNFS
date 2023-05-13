#pragma once

#include <iostream>
#include <random>
#include "Layer.hpp"

namespace NNFSCore
{

    /**
     * @brief Dense layer
     *
     * @details This class implements the dense layer. It is the most basic layer in a neural network.
     */
    class Dense : public Layer
    {
    public:
        /**
         * @brief Construct a new Dense object.
         *
         * @details This constructor initializes the weights and biases of the layer. It also sets all regularization matrices to zero.
         *
         * @param n_input Number of input neurons
         * @param n_output Number of output neurons
         * @param l1_weights_regularizer L1 regularization for weights (default: 0.0)
         * @param l1_biases_regularizer L1 regularization for biases (default: 0.0)
         * @param l2_weights_regularizer L2 regularization for weights (default: 0.0)
         * @param l2_biases_regularizer L2 regularization for biases (default: 0.0)
         */
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

        /**
         * @brief Forward pass of the dense layer
         *
         * @param[out] out Output of the layer
         * @param[in] x Input of the layer
         */
        void forward(Eigen::MatrixXd &out, const Eigen::MatrixXd &x)
        {
            _forward_input = x;
            out = x * _weights;
            for (int row = 0; row < out.rows(); ++row)
            {
                out.row(row).array() += _biases.array();
            }
        }

        /**
         * @brief Backward pass of the dense layer
         *
         * @param[out] out Input gradient
         * @param[in] dx Output gradient
         */
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

        /**
         * @brief Get weights
         *
         * @return Eigen::MatrixXd& Weights
         */
        const Eigen::MatrixXd &weights() const
        {
            return _weights;
        }

        /**
         * @brief Get biases
         *
         * @return Eigen::MatrixXd& Biases
         */
        const Eigen::MatrixXd &biases() const
        {
            return _biases;
        }

        /**
         * @brief Get weights gradients
         *
         * @return Eigen::MatrixXd& Weights gradients
         */
        const Eigen::MatrixXd &dweights() const
        {
            return _dweights;
        }

        /**
         * @brief Get biases gradients
         *
         * @return Eigen::MatrixXd& Biases gradients
         */
        const Eigen::MatrixXd &dbiases() const
        {
            return _dbiases;
        }

        /**
         * @brief Set's the weights of the dense layer
         *
         * @details This function sets the weights of the dense layer. The shape of the new weights matrix must match the shape of the initial weights matrix.
         *
         * @param[in] weights New weights matrix
         *
         * @throws std::invalid_argument if the shape of the new weights matrix does not match the shape of the initial weights matrix.
         */
        void weights(Eigen::MatrixXd &weights)
        {
            if (_weights.rows() != weights.rows() || _weights.cols() != weights.cols())
            {
                LOG_ERROR("Shape of new matrix does not match to initial's matrix shape.");
                throw std::invalid_argument("Shape of new matrix does not match to initial's matrix shape.");
            }

            _weights = weights;
        }

        /**
         * @brief Set's the biases of the dense layer
         *
         * @details This function sets the biases of the dense layer. The shape of the new biases matrix must match the shape of the initial biases matrix.
         *
         * @param[in] biases New biases matrix
         *
         * @throws std::invalid_argument if the shape of the new biases matrix does not match the shape of the initial biases matrix.
         */
        void biases(Eigen::MatrixXd &biases)
        {
            if (_biases.rows() != biases.rows() || _biases.cols() != biases.cols())
            {
                LOG_ERROR("Shape of new matrix does not match to initial's matrix shape.");
                throw std::invalid_argument("Shape of new matrix does not match to initial's matrix shape.");
            }
            _biases = biases;
        }

        /**
         * @brief Set's the weights optimizer matrix of the dense layer
         *
         * @param woptimizer New weights optimizer matrix
         */
        void weights_optimizer(Eigen::MatrixXd woptimizer)
        {
            _weights_optimizer = woptimizer;
        }

        /**
         * @brief Set's the biases optimizer matrix of the dense layer
         *
         * @param boptimizer New biases optimizer matrix
         */
        void biases_optimizer(Eigen::MatrixXd boptimizer)
        {
            _biases_optimizer = boptimizer;
        }

        /**
         * @brief Get weights optimizer matrix
         *
         * @return Eigen::MatrixXd& Weights optimizer matrix
         */
        const Eigen::MatrixXd &weights_optimizer() const
        {
            return _weights_optimizer;
        }

        /**
         * @brief Get biases optimizer matrix
         *
         * @return Eigen::MatrixXd& Biases optimizer matrix
         */
        const Eigen::MatrixXd &biases_optimizer() const
        {
            return _biases_optimizer;
        }

        /**
         * @brief Set's the additional weights optimizer matrix of the dense layer
         *
         * @param woptimizer New additional weights optimizer matrix
         */
        void weights_optimizer_additional(Eigen::MatrixXd woptimizer)
        {
            _weights_optimizer_additional = woptimizer;
        }

        /**
         * @brief Set's the additional biases optimizer matrix of the dense layer
         *
         * @param boptimizer New additional biases optimizer matrix
         */
        void biases_optimizer_additional(Eigen::MatrixXd boptimizer)
        {
            _biases_optimizer_additional = boptimizer;
        }

        /**
         * @brief Get additional weights optimizer matrix
         *
         * @return Eigen::MatrixXd& Additional weights optimizer matrix
         */
        const Eigen::MatrixXd &weights_optimizer_additional() const
        {
            return _weights_optimizer_additional;
        }

        /**
         * @brief Get additional biases optimizer matrix
         *
         * @return Eigen::MatrixXd& Additional biases optimizer matrix
         */
        const Eigen::MatrixXd &biases_optimizer_additional() const
        {
            return _biases_optimizer_additional;
        }

        /**
         * @brief Get L1 weights regularizer
         *
         * @return double L1 weights regularizer
         */
        const double &l1_weights_regularizer() const
        {
            return _l1_weights_regularizer;
        }

        /**
         * @brief Get L2 weights regularizer
         *
         * @return double L2 weights regularizer
         */
        const double &l2_weights_regularizer() const
        {
            return _l2_weights_regularizer;
        }

        /**
         * @brief Get L1 biases regularizer
         *
         * @return double L1 biases regularizer
         */
        const double &l1_biases_regularizer() const
        {
            return _l1_biases_regularizer;
        }

        /**
         * @brief Get L2 biases regularizer
         *
         * @return double L2 biases regularizer
         */
        const double &l2_biases_regularizer() const
        {
            return _l2_biases_regularizer;
        }

        /**
         * @brief Calculates the number of trainable of the dense layer
         *
         * @return int Number of parameters
         */
        int parameters() const
        {
            return _n_input * _n_output + _n_output;
        }

        /**
         * @brief Gives the shape of the dense layer
         *
         * @param[out] n_input Number of input neurons
         * @param[out] n_output Number of output neurons
         */
        void shape(int &n_input, int &n_output) const
        {
            n_input = _n_input;
            n_output = _n_output;
        }

    private:
        int _n_input;  // Number of input neurons
        int _n_output; // Number of output neurons

        Eigen::MatrixXd _weights; // Weights matrix
        Eigen::MatrixXd _biases;  // Biases matrix

        Eigen::MatrixXd _dweights; // Weights gradients
        Eigen::MatrixXd _dbiases;  // Biases gradients

        Eigen::MatrixXd _weights_optimizer; // Weights optimizer matrix
        Eigen::MatrixXd _biases_optimizer;  // Biases optimizer matrix

        Eigen::MatrixXd _weights_optimizer_additional; // Additional weights optimizer matrix
        Eigen::MatrixXd _biases_optimizer_additional;  // Additional biases optimizer matrix

        double _l1_weights_regularizer; // L1 weights regularizer
        double _l1_biases_regularizer;  // L1 biases regularizer
        double _l2_weights_regularizer; // L2 weights regularizer
        double _l2_biases_regularizer;  // L2 biases regularizer

        Eigen::MatrixXd _forward_input; // Forward input
    };
} // namespace NNFSCore
