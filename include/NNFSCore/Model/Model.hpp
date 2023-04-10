#ifndef MODEL_HPP
#define MODEL_HPP

#include <Eigen/Dense>
#include "../Activation/Activation.hpp"
#include "../Callback/Callback.hpp"
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

        /**
         * @brief Get the learning rate
         * @return The learning rate
         */
        virtual double get_learning_rate() const = 0;

        /**
         * @brief Set the learning rate
         * @param value The new learning rate
         */
        virtual void set_learning_rate(double value) = 0;

        /**
         * @brief Compute the output of the model given an input tensor
         * @param input_tensor Input tensor
         * @return The output tensor
         */
        virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) = 0;

        /**
         * @brief Fit the model to the given data
         * @param examples Input examples
         * @param labels Input labels
         * @param epochs Number of epochs
         * @param callbacks A vector of pointers to Callback objects.
         */
        virtual void fit(const Eigen::MatrixXd &examples, const Eigen::MatrixXd &labels, int epochs,
                         bool verbose = false, const std::vector<std::shared_ptr<Callback>> &callbacks = {}) = 0;

        /**
         * @brief Predict the output for given examples
         * @param examples Input examples
         * @return Predicted output
         */
        virtual Eigen::MatrixXd predict(const Eigen::MatrixXd &examples) = 0;

        /**
         * @brief Evaluate the model for given examples and labels
         * @param examples Input examples
         * @param labels Input labels
         * @return Evaluation result
         */
        virtual Eigen::MatrixXd evaluate(const Eigen::MatrixXd &examples,
                                         const Eigen::MatrixXd &labels) = 0;

        /**
         * @brief Perform a backward step for the model
         * @param labels Input labels
         */
        virtual void backward_step(const Eigen::MatrixXd &labels) = 0;

        /**
         * @brief Get the layers of the network.
         *
         * @return std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> The layers.
         */
        virtual std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> get_layers() const = 0;

        /**
         * @brief Set the callbacks for the model.
         *
         * @param callbacks A vector of pointers to Callback objects.
         */
        virtual void set_callbacks(const std::vector<std::shared_ptr<Callback>> &callbacks) = 0;

        /**
         * @brief Get the callbacks for the model.
         *
         * @return std::vector<std::shared_ptr<Callback>> The callbacks.
         */
        virtual std::vector<std::shared_ptr<Callback>> get_callbacks() const = 0;

        /**
         * @brief Get the number of layers in the network.
         *
         * @return int The number of layers.
         */
        virtual int get_num_layers() const = 0;

        /**
         * @brief Set the number of layers in the network.
         *
         * @param num_layers The number of layers.
         */
        virtual void set_num_layers(int num_layers) = 0;

        /**
         * @brief Get the number of examples used for training the model.
         *
         * @return int The number of examples.
         */
        virtual int get_num_examples() const = 0;

        /**
         * @brief Set the number of examples used for training the model.
         *
         * @param num_examples The number of examples.
         */
        virtual void set_num_examples(int num_examples) = 0;

        /**
         * @brief Update the model's parameters
         */
        virtual void update() = 0;
    };

} // namespace NNFSCore

#endif // MODEL_HPP
