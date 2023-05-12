#pragma once

#include "CCE.hpp"
#include "../Activation/Softmax.hpp"

namespace NNFSCore
{
    class CCESoftmax : public Loss
    {
    public:
        CCESoftmax(std::shared_ptr<Softmax> softmax, std::shared_ptr<CCE> cce) : Loss(LossType::CCE_SOFTMAX), _softmax(softmax), _cce(cce) {}

        void forward(Eigen::MatrixXd &sample_losses, const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) const
        {
            Eigen::MatrixXd out;

            _softmax->forward(out, predictions);

            _cce->forward(sample_losses, out, labels);
        }

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

        Eigen::MatrixXd &softmax_out() const
        {
            return _softmax->_forward_output;
        }

    private:
        std::shared_ptr<Softmax> _softmax;
        std::shared_ptr<CCE> _cce;
        Eigen::MatrixXd _softmax_out;
    };
}

