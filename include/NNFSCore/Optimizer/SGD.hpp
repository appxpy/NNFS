#pragma once

#include <Eigen/Dense>
#include "Optimizer.hpp"

namespace NNFSCore
{
    class SGD : public Optimizer
    {
    public:
        SGD(double lr, double decay = 0.0, double momentum = 0.0) : Optimizer(lr),
                                                                    _decay(decay),
                                                                    _iterations(0),
                                                                    _momentum(momentum) {}

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

        void pre_update_params()
        {
            if (_decay > 0)
            {
                _current_lr = _lr * (1. / (1. + _decay * _iterations));
            }
        }

        void post_update_params()
        {
            _iterations += 1;
        }

    private:
        double _decay;
        int _iterations;
        double _momentum;
    };
}

