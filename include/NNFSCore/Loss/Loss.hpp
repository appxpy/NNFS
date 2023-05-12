#pragma once

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"
#include "../Layer/Dense.hpp"

namespace NNFSCore
{
    enum class LossType
    {
        CCE,
        CCE_SOFTMAX
    };

    class Loss
    {
    public:
        LossType type;

    public:
        Loss(LossType type) : type(type) {}

        virtual ~Loss() = default;

        virtual void forward(Eigen::MatrixXd &sample_losses, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const = 0;

        virtual void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const = 0;

        void calculate(double &loss, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels)
        {
            Eigen::MatrixXd sample_losses;
            forward(sample_losses, predictions, labels);
            loss = sample_losses.mean();
        }

        double regularization_loss(const std::shared_ptr<Dense> &layer)
        {
            double regularization_loss = 0;
            const double weight_regularizer_l1 = layer->l1_weights_regularizer();
            const double weight_regularizer_l2 = layer->l2_weights_regularizer();
            const double bias_regularizer_l1 = layer->l1_biases_regularizer();
            const double bias_regularizer_l2 = layer->l2_biases_regularizer();

            if (weight_regularizer_l1 > 0)
            {
                regularization_loss += weight_regularizer_l1 * layer->weights().array().abs().sum();
            }

            if (weight_regularizer_l2 > 0)
            {
                regularization_loss += weight_regularizer_l2 * (layer->weights().array() * layer->weights().array()).sum();
            }

            if (bias_regularizer_l1 > 0)
            {
                regularization_loss += bias_regularizer_l1 * layer->weights().array().abs().sum();
            }

            if (bias_regularizer_l2 > 0)
            {
                regularization_loss += bias_regularizer_l2 * (layer->biases().array() * layer->biases().array()).sum();
            }

            return regularization_loss;
        }
    };
}
