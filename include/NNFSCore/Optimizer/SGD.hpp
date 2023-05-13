#pragma once

#include <Eigen/Dense>
#include "Optimizer.hpp"

namespace NNFSCore
{
    /**
     * @brief Stochastic Gradient Descent optimizer
     *
     * @details This class implements the Stochastic Gradient Descent optimizer.
     */
    class SGD : public Optimizer
    {
    public:
        /**
         * @brief Construct a new SGD object
         *
         * @param lr Learning rate
         * @param decay Learning rate decay (default: 0.0)
         * @param momentum Momentum (default: 0.0)
         */
        SGD(double lr, double decay = 0.0, double momentum = 0.0) : Optimizer(lr),
                                                                    _decay(decay),
                                                                    _iterations(0),
                                                                    _momentum(momentum) {}

        /**
         * @brief Update the parameters of the layer
         *
         * @param[in,out] layer Layer to update
         */
        void update_params(std::shared_ptr<Dense> &layer)
        {
            Eigen::MatrixXd weights = layer->weights();
            Eigen::MatrixXd biases = layer->biases();
            Eigen::MatrixXd dweights = layer->dweights();
            Eigen::MatrixXd dbiases = layer->dbiases();

            Eigen::MatrixXd weights_updates;
            Eigen::MatrixXd bias_updates;
            if (_momentum > 0)
            {
                Eigen::MatrixXd weights_momentums = layer->weights_optimizer();
                Eigen::MatrixXd biases_momentums = layer->biases_optimizer();

                weights_updates = _momentum * weights_momentums - _current_lr * dweights;

                bias_updates = _momentum * biases_momentums - _current_lr * dbiases;

                layer->weights_optimizer(weights_updates);
                layer->biases_optimizer(bias_updates);
            }
            else
            {
                weights_updates = -_current_lr * dweights;
                bias_updates = -_current_lr * dbiases;
            }

            weights += weights_updates;
            biases += bias_updates;

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
        double _decay;    // Learning rate decay
        int _iterations;  // Iteration count
        double _momentum; // Momentum
    };
} // namespace NNFSCore
