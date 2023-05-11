#ifndef ACCURACY_METRICS_HPP
#define ACCURACY_METRICS_HPP

#include <Eigen/Dense>
#include "../Utilities/clue.hpp"

namespace NNFSCore
{

    class Metrics
    {
    public:
        static void accuracy(double &accuracy, const Eigen::MatrixXd &predicted,
                             const Eigen::MatrixXd &labels)
        {
            Eigen::VectorXi absolute_predictions(predicted.rows());
            for (int i = 0; i < predicted.rows(); ++i)
            {
                predicted.row(i).maxCoeff(&absolute_predictions[i]);
            };

            Eigen::VectorXi class_labels(labels.rows());
            for (int k = 0; k < labels.rows(); ++k)
            {
                labels.row(k).maxCoeff(&class_labels[k]);
            };

            accuracy = (absolute_predictions.array() == class_labels.array()).cast<double>().mean();
        }
    };
};

#endif