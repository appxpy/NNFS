#pragma once

#include <Eigen/Dense>
#include "Optimizer.hpp"

namespace NNFSCore
{
    /**
     * @brief Adagrad optimizer (Adaptive Gradient)
     *
     * @details This class implements the Adagrad optimizer.
     */
    class Adagrad : public Optimizer
    {
    public:
        /**
         * @brief Construct a new Adagrad object
         *
         * @param lr Learning rate
         * @param decay Learning rate decay (default: 0.0)
         * @param epsilon Epsilon value to avoid division by zero (default: 1e-7)
         */
        Adagrad(double lr, double decay = 0.0, double epsilon = 1e-7) : Optimizer(lr),
                                                                        _decay(decay),
                                                                        _iterations(0),
                                                                        _epsilon(epsilon) {}

        /**
         * @brief Update the parameters of the layer
         *
         * @param[in,out] layer Layer to update
         *
         */
        void update_params(std::shared_ptr<Dense> &layer)
        {
            Eigen::MatrixXd weights = layer->weights();
            Eigen::MatrixXd biases = layer->biases();
            Eigen::MatrixXd dweights = layer->dweights();
            Eigen::MatrixXd dbiases = layer->dbiases();

            Eigen::MatrixXd weights_cache = layer->weights_optimizer();
            Eigen::MatrixXd biases_cache = layer->biases_optimizer();

            weights_cache += dweights.cwisePow(2);
            biases_cache += dbiases.cwisePow(2);

            weights += (-_current_lr * dweights.array() / (weights_cache.array().sqrt() + _epsilon)).matrix();
            biases += (-_current_lr * dbiases.array() / (biases_cache.array().sqrt() + _epsilon)).matrix();

            layer->weights_optimizer(weights_cache);
            layer->biases_optimizer(biases_cache);

            layer->weights(weights);
            layer->biases(biases);
        }

        /**
         * @brief Pre-update parameters (e.g. learning rate decay)
         */
        void pre_update_params()
        {
            if (_decay > 0)
            {
                _current_lr = _lr * (1. / (1. + _decay * _iterations));
            }
        }

        /**
         * @brief Post-update parameters (e.g. increase iteration count)
         */
        void post_update_params()
        {
            _iterations += 1;
        }

    private:
        double _decay;   // Learning rate decay
        int _iterations; // Iteration count
        double _epsilon; // Epsilon - to avoid division by zero
    };
} // namespace NNFSCore
