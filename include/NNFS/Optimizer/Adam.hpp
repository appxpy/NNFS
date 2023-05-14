#pragma once

#include "Optimizer.hpp"

namespace NNFS
{
    /**
     * @brief Adam optimizer - Adaptive Moment Estimation, one of the most popular and efficient gradient-based optimization algorithms
     *
     * @details This class implements the Adam optimizer.
     */
    class Adam : Optimizer
    {
    public:
        /**
         * @brief Construct a new Adam object
         *
         * @param lr Learning rate (default: 1e-3)
         * @param decay Learning rate decay (default: 0.0)
         * @param epsilon Epsilon value to avoid division by zero (default: 1e-7)
         * @param beta_1 Exponential decay rate for the first moment estimates (default: 0.9)
         * @param beta_2 Exponential decay rate for the second moment estimates (default: 0.999)
         */
        Adam(double lr = 1e-3, double decay = .0, double epsilon = 1e-7, double beta_1 = .9, double beta_2 = .999) : Optimizer(lr, decay),
                                                                                                                     _epsilon(epsilon),
                                                                                                                     _beta_1(beta_1),
                                                                                                                     _beta_2(beta_2) {}

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

            Eigen::MatrixXd weights_momentums = layer->weights_optimizer_additional();
            Eigen::MatrixXd biases_momentums = layer->biases_optimizer_additional();

            weights_momentums = _beta_1 * weights_momentums + (1 - _beta_1) * dweights;
            biases_momentums = _beta_1 * biases_momentums + (1 - _beta_1) * dbiases;

            Eigen::MatrixXd weights_momentums_corrected = weights_momentums / (1 - std::pow(_beta_1, (_iterations + 1)));
            Eigen::MatrixXd biases_momentums_corrected = biases_momentums / (1 - std::pow(_beta_1, (_iterations + 1)));

            weights_cache = _beta_2 * weights_cache + (1 - _beta_2) * dweights.cwisePow(2);
            biases_cache = _beta_2 * biases_cache + (1 - _beta_2) * dbiases.cwisePow(2);

            Eigen::MatrixXd weights_cache_corrected = weights_cache / (1 - std::pow(_beta_2, (_iterations + 1)));
            Eigen::MatrixXd biases_cache_corrected = biases_cache / (1 - std::pow(_beta_2, (_iterations + 1)));

            weights += ((-_current_lr * weights_momentums_corrected).array() / (weights_cache_corrected.cwisePow(.5).array() + _epsilon)).matrix();
            biases += ((-_current_lr * biases_momentums_corrected).array() / (biases_cache_corrected.cwisePow(.5).array() + _epsilon)).matrix();

            layer->weights_optimizer(weights_cache);
            layer->biases_optimizer(biases_cache);

            layer->weights_optimizer_additional(weights_momentums);
            layer->biases_optimizer_additional(biases_momentums);

            layer->weights(weights);
            layer->biases(biases);
        }

    private:
        double _epsilon; // Epsilon value to avoid division by zero
        double _beta_1;  // Exponential decay rate for the first moment estimates
        double _beta_2;  // Exponential decay rate for the second moment estimates
    };
} // namespace NNFS
