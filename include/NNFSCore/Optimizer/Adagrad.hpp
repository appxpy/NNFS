#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

#include <Eigen/Dense>
#include "Optimizer.hpp"

namespace NNFSCore
{
    class Adagrad : public Optimizer
    {
    public:
        Adagrad(double lr, double decay = 0.0, double epsilon = 1e-7) : Optimizer(lr),
                                                                        _decay(decay),
                                                                        _iterations(0),
                                                                        _epsilon(epsilon) {}

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
        double _epsilon;
    };
}

#endif