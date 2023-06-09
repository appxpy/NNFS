#pragma once

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"

namespace NNFS
{
    /**
     * @brief Metrics class
     *
     * @details This class implements the metrics used to evaluate the performance of the model.
     */
    class Metrics
    {
    public:
        /**
         * @brief Calculates the accuracy of the model.
         *
         * @param[out] accuracy The accuracy of the model.
         * @param[in] predicted The predicted data.
         * @param[in] labels The labels.
         */
        static void accuracy(double &accuracy, const Eigen::MatrixXd &predicted,
                             const Eigen::MatrixXd &labels)
        {
            Eigen::VectorXi absolute_predictions;
            onehotdecode(absolute_predictions, predicted);

            Eigen::VectorXi class_labels;
            onehotdecode(class_labels, labels);

            accuracy = (absolute_predictions.array() == class_labels.array()).cast<double>().mean();
        }

        /**
         * @brief Decodes one-hot encoded data.
         *
         * @param[out] decoded The decoded data.
         * @param[in] onehot The one-hot encoded data.
         */
        static void onehotdecode(Eigen::VectorXi &decoded, const Eigen::MatrixXd &onehot)
        {
            decoded.resize(onehot.rows());
            for (int i = 0; i < onehot.rows(); ++i)
            {
                onehot.row(i).maxCoeff(&decoded[i]);
            };
        }
    };
} // namespace NNFS
