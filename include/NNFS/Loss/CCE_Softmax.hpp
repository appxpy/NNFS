#pragma once

#include "CCE.hpp"
#include "../Activation/Softmax.hpp"

namespace NNFS
{
    /**
     * @brief Cross-entropy loss function with softmax activation
     *
     * @details This class implements the cross-entropy loss function with softmax activation.
     */
    class CCESoftmax : public Loss
    {
    public:
        /**
         * @brief Construct a new CCESoftmax object
         *
         * @param softmax Softmax activation layer
         * @param cce Cross-entropy loss function
         */
        CCESoftmax(std::shared_ptr<Softmax> softmax, std::shared_ptr<CCE> cce) : Loss(LossType::CCE_SOFTMAX), _softmax(softmax), _cce(cce) {}

        /**
         * @brief Forward pass of the CCE loss function with softmax activation
         *
         * @param[out] sample_losses Sample losses
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        void forward(Eigen::MatrixXd &sample_losses, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            Eigen::MatrixXd out;

            _softmax->forward(out, predictions);

            _cce->forward(sample_losses, out, labels);
        }

        /**
         * @brief Backward pass of the CCE loss function with softmax activation
         *
         * @param[out] out Output gradient
         * @param[in] predictions Predictions
         * @param[in] labels Labels
         */
        void backward(Eigen::MatrixXd &out, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            Eigen::MatrixXd _predictions = softmax_out();
            int samples = predictions.rows();

            Eigen::VectorXi class_labels(labels.rows());
            for (int k = 0; k < labels.rows(); ++k)
            {
                labels.row(k).maxCoeff(&class_labels[k]);
            };

            out = _predictions;

            // Calculate gradient
            for (int i = 0; i < samples; i++)
            {
                int index = class_labels(i);
                out(i, index) -= 1;
            }

            // Normalize gradient
            out /= samples;
        }

        /**
         * @brief Get the softmax output
         *
         * @return Eigen::MatrixXd& Softmax output
         */
        Eigen::MatrixXd &softmax_out() const
        {
            return _softmax->_forward_output;
        }

    private:
        std::shared_ptr<Softmax> _softmax; // Softmax activation layer
        std::shared_ptr<CCE> _cce;         // Cross-entropy loss function
        Eigen::MatrixXd _softmax_out;      // Softmax output
    };
} // namespace NNFS
