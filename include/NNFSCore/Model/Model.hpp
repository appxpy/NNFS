#pragma once

#include <Eigen/Dense>

#include "../Layer/Layer.hpp"
#include "../Loss/Loss.hpp"

namespace NNFSCore
{

    /**
     * @class Model
     * @brief Abstract base class for the model in a neural network
     */
    class Model
    {
    public:
        virtual ~Model() = default;

        virtual void fit(const Eigen::MatrixXd &examples, const Eigen::MatrixXd &labels, const Eigen::MatrixXd &test_examples, const Eigen::MatrixXd &test_labels, int epochs, int batch_size, bool verbose = false) = 0;

    private:
        virtual void backward(Eigen::MatrixXd &predicted, const Eigen::MatrixXd &labels) = 0;

        virtual void forward(Eigen::MatrixXd &x) = 0;
    };

} // namespace NNFSCore
