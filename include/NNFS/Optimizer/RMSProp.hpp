#pragma once

#include <Eigen/Dense>
#include "Optimizer.hpp"

namespace NNFS
{
    /**
     * @brief Root Mean Square Propagation optimizer
     *
     * @details This class implements the Root Mean Square Propagation (RMSProp) optimizer.
     */
    class RMSProp : Optimizer
    {
    public:
        /**
         * @brief Construct a new RMSProp object
         *
         * @param lr Learning rate (default: 1e-3)
         * @param decay Learning rate decay  (default: 1e-3)
         * @param epsilon Epsilon - to avoid division by zero (default: 1e-7)
         * @param rho RMSProp uses "rho" to calculate an exponentially weighted average over the square of the gradients. (default: .9)
         */
        RMSProp(double lr = 1e-3, double decay = 1e-3, double epsilon = 1e-7, double rho = .9) : Optimizer(lr, decay),
                                                                                                 _epsilon(epsilon),
                                                                                                 _rho(rho)
        {
        }

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

    private:
        double _epsilon; // Epsilon - to avoid division by zero
        double _rho;     // RMSProp uses "rho" to calculate an exponentially weighted average over the square of the gradients.
    };
} // namespace NNFS
