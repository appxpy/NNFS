#ifndef RMSPROP_HPP
#define RMSPROP_HPP

#include <Eigen/Dense>
#include "Optimizer.hpp"

namespace NNFSCore
{
    class RMSProp : public Optimizer
    {
    public:
        RMSProp(double lr = 1e-3, double decay = 1e-3, double epsilon = 1e-7, double rho = .9) : Optimizer(lr),
                                                                                                 _decay(decay),
                                                                                                 _iterations(0),
                                                                                                 _epsilon(epsilon),
                                                                                                 _rho(rho) {}

        void update_params(std::shared_ptr<Dense> &layer)
        {
            Eigen::MatrixXd weights = layer->weights();
            Eigen::MatrixXd biases = layer->biases();
            Eigen::MatrixXd dweights = layer->dweights();
            Eigen::MatrixXd dbiases = layer->dbiases();

            Eigen::MatrixXd weights_cache = layer->weights_optimizer();
            Eigen::MatrixXd biases_cache = layer->biases_optimizer();

            weights_cache = _rho * weights_cache + (1 - _rho) * dweights.cwisePow(2);
            biases_cache = _rho * biases_cache + (1 - _rho) * dbiases.cwisePow(2);

            weights += (-_current_lr * dweights.array() / (weights_cache.cwisePow(.5).array() + _epsilon)).matrix();
            biases += (-_current_lr * dbiases.array() / (biases_cache.cwisePow(.5).array() + _epsilon)).matrix();

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
        double _rho;
    };
}

#endif