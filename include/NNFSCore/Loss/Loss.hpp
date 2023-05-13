#pragma once

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"
#include "../Layer/Dense.hpp"

namespace NNFSCore
{
    /**
     * @brief Enum class for loss types
     */
    enum class LossType
    {
        CCE,
        CCE_SOFTMAX
    };

    /**
     * @brief Base class for all loss functions
     *
     * @details This class is the base class for all losses. It provides the interface for all loss functions.
     */
    class Loss
    {
    public:
        LossType type; // Type of loss function

    public:
        /**
         * @brief Construct a new Loss object
         *
         * @param type Type of loss function
         */
        Loss(LossType type) : type(type) {}

        /**
         * @brief Basic destructor
         */
        virtual ~Loss() = default;

        /**
         * @brief Forward pass of the loss function
         *
         * @param[out] sample_losses Sample losses
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        virtual void forward(Eigen::MatrixXd &sample_losses, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const = 0;

        /**
         * @brief Backward pass of the loss function
         *
         * @param[out] out Output gradient
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        virtual void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const = 0;

        /**
         * @brief Calculate the loss
         *
         * @param[out] loss Loss
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        void calculate(double &loss, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels)
        {
            Eigen::MatrixXd sample_losses;
            forward(sample_losses, predictions, labels);
            loss = sample_losses.mean();
        }

        /**
         * @brief Calculate l1 and l2 regularization loss.
         *
         * @param layer Layer to calculate regularization loss
         *
         * @return double Regularization loss
         */
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
} // namespace NNFSCore
