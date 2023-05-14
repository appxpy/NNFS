#pragma once

#include <Eigen/Dense>

#include "../Layer/Layer.hpp"
#include "../Loss/Loss.hpp"

namespace NNFS
{

    /**
     * @brief Abstract base class for the model in a neural network
     *
     * @details This class is the abstract base class for the model in a neural network. It provides the interface for all models.
     *
     * @todo Add support for callbacks
     */
    class Model
    {
    public:
        /**
         * @brief Basic destructor
         */
        virtual ~Model() = default;

        /**
         * @brief Evaluate the model on the given examples
         *
         * @param[in] examples Examples to evaluate the model on
         * @param[in] labels Labels of the examples
         * @param[in] test_examples Examples to validate the model on
         * @param[in] test_labels Labels of the validation examples
         * @param[in] epochs The number of epochs to train for
         * @param[in] batch_size The batch size, i.e. the number of examples to train on in each batch
         * @param[in] verbose Whether to print out information about the training process
         *
         * @note This is a pure virtual function and must be implemented by the derived class.
         */
        virtual void fit(const Eigen::MatrixXd &examples, const Eigen::MatrixXd &labels, const Eigen::MatrixXd &test_examples, const Eigen::MatrixXd &test_labels, int epochs, int batch_size, bool verbose = false) = 0;

    private:
        /**
         * @brief Implements the backward pass of the neural network.
         *
         * @details The backward pass of the neural network is implemented by first calculating the loss of the neural network and then propagating the loss backwards through the neural network.
         *
         * @param[in] predicted Predicted output of the neural network.
         * @param[in] labels Labels of the provided examples.
         *
         * @note This is a pure virtual function and must be implemented by the derived class.
         */
        virtual void backward(Eigen::MatrixXd &predicted, const Eigen::MatrixXd &labels) = 0;

        /**
         * @brief Implements the forward pass of the neural network.
         *
         * @param[in, out] x Input of the neural network. The output of the neural network is stored in this matrix.
         *
         * @note This is a pure virtual function and must be implemented by the derived class.
         */
        virtual void forward(Eigen::MatrixXd &x) = 0;
    };

} // namespace NNFS
