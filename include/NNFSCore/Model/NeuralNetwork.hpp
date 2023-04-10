#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <tuple>
#include <vector>

#include "Model.hpp"
#include "../Layer/Layer.hpp"
#include "../Activation/Activation.hpp"
#include "../Loss/Loss.hpp"

namespace NNFSCore
{

    /**
     * @class NeuralNetwork
     * @brief A class implementing a neural network model
     */
    class NeuralNetwork : public Model
    {
    public:
        /**
         * @brief Constructor for NeuralNetwork
         * @param layers Vector of layer-activation tuples
         * @param loss Shared pointer to the loss function
         * @param learning_rate The learning rate of the model
         * @param regularization_factor The regularization factor of the model
         */
        NeuralNetwork(
            std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> &layers,
            std::shared_ptr<Loss> loss, double learning_rate,
            double regularization_factor)
            : _layers(layers),
              _loss(loss),
              _learning_rate(learning_rate),
              _regularization_factor(regularization_factor),
              _input(),
              _output(),
              _num_layers(layers.size()),
              _num_examples(0)
        {
        }

        /**
         * @brief Get the learning rate
         * @return The learning rate
         */
        double get_learning_rate() const override
        {
            return _learning_rate;
        }

        /**
         * @brief Set the learning rate
         * @param value The new learning rate
         */
        void set_learning_rate(double value) override
        {
            _learning_rate = value;
        }

        /**
         * @brief Get the number of layers in the network.
         *
         * @return int The number of layers.
         */
        int get_num_layers() const override
        {
            return _num_layers;
        }

        /**
         * @brief Set the number of layers in the network.
         *
         * @param num_layers The number of layers.
         */
        void set_num_layers(int num_layers) override
        {
            _num_layers = num_layers;
        }

        /**
         * @brief Get the number of examples used for training the model.
         *
         * @return int The number of examples.
         */
        int get_num_examples() const override
        {
            return _num_examples;
        }

        /**
         * @brief Set the number of examples used for training the model.
         *
         * @param num_examples The number of examples.
         */
        void set_num_examples(int num_examples) override
        {
            _num_examples = num_examples;
        }

        /**
         * @brief Compute the output of the model given an input tensor
         * @param input_tensor Input tensor
         * @return The output tensor
         */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd &input_tensor) override
        {
            if (_num_examples == 0)
            {
                _num_examples = input_tensor.cols();
            }

            _input = input_tensor;

            return forward_pass(input_tensor);
        }

        /**
         * @brief Set the callbacks for the model.
         *
         * @param callbacks A vector of pointers to Callback objects.
         */
        void set_callbacks(const std::vector<std::shared_ptr<Callback>> &callbacks) override
        {
            _callbacks = callbacks;
        }

        /**
         * @brief Get the callbacks for the model.
         *
         * @return std::vector<std::shared_ptr<Callback>> The callbacks.
         */
        std::vector<std::shared_ptr<Callback>> get_callbacks() const override
        {
            return _callbacks;
        }

        /**
         * @brief Fit the model to a set of examples and labels.
         *
         * @param examples The examples tensor.
         * @param labels The labels tensor.
         * @param epochs The number of epochs to train for.
         * @param verbose Whether to print progress during training.
         * @param callbacks A vector of pointers to Callback objects.
         */
        void fit(const Eigen::MatrixXd &examples, const Eigen::MatrixXd &labels, int epochs,
                 bool verbose = false, const std::vector<std::shared_ptr<Callback>> &callbacks = {}) override
        {
            _callbacks = callbacks;

            for (int epoch = 1; epoch <= epochs; ++epoch)
            {
                (*this)(examples);
                Eigen::MatrixXd loss = _loss->operator()(_output, labels);

                backward_pass(labels);
                update();

                for (auto &callback : _callbacks)
                {
                    callback->on_epoch_end(epoch, loss(0));
                }

                if (verbose)
                {
                    std::cout << "Epoch: " << epoch << ", Loss: " << loss(0) << std::endl;
                }
            }
        }

        /**
         * @brief Predict the output for given examples
         * @param examples Input examples
         * @return Predicted output
         */
        Eigen::MatrixXd
        predict(const Eigen::MatrixXd &examples) override
        {
            Eigen::MatrixXd outputs = (*this)(examples);
            return (outputs.array() > 0.5).cast<double>();
        }

        /**
         * @brief Evaluate the model for given examples and labels
         * @param examples Input examples
         * @param labels Input labels
         * @return Evaluation result
         */
        Eigen::MatrixXd evaluate(const Eigen::MatrixXd &examples,
                                 const Eigen::MatrixXd &labels) override
        {
            (*this)(examples);
            return _loss->operator()(_output, labels);
        }

        /**
         * @brief Get the layers of the network.
         *
         * @return std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> The layers.
         */
        std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> get_layers() const override
        {
            return _layers;
        }

        /**
         * @brief Get the loss function used by the network.
         *
         * @return std::shared_ptr<Loss> The loss function.
         */
        std::shared_ptr<Loss> get_loss() const
        {
            return _loss;
        }

        /**
         * @brief Get the regularization factor used by the network.
         *
         * @return double The regularization factor.
         */
        double get_regularization_factor() const
        {
            return _regularization_factor;
        }

        /**
         * @brief Perform a backward step for the model
         * @param labels Input labels
         */
        void backward_step(const Eigen::MatrixXd &labels) override
        {
            backward_pass(labels);
        }

        /**
         * @brief Update the model's parameters
         */
        void update() override
        {
            for (auto &layer : _layers)
            {
                auto &layer_ptr = std::get<0>(layer);
                layer_ptr->update(_learning_rate);
            }
        }

    private:
        std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> _layers;
        std::shared_ptr<Loss> _loss;
        double _learning_rate;
        double _regularization_factor;
        Eigen::MatrixXd _input;
        Eigen::MatrixXd _output;
        int _num_layers;
        int _num_examples;
        std::vector<std::shared_ptr<Callback>> _callbacks;

        /**
         * @brief Perform a forward pass through the network
         * @param input_tensor Input tensor
         * @return The output tensor
         */
        Eigen::MatrixXd forward_pass(const Eigen::MatrixXd &input_tensor)
        {
            Eigen::MatrixXd output = input_tensor;

            for (auto &layer : _layers)
            {
                auto &layer_ptr = std::get<0>(layer);
                auto &activation_ptr = std::get<1>(layer);

                output = (*layer_ptr)(output);
                output = (*activation_ptr)(output);
            }

            _output = output;
            return _output;
        }

        /**
         * @brief Perform a backward pass through the network
         * @param labels Input labels
         */
        void backward_pass(const Eigen::MatrixXd &labels)
        {
            Eigen::MatrixXd da = _loss->gradient(_output, labels);

            for (int i = _num_layers - 1; i >= 0; --i)
            {
                auto &layer = std::get<0>(_layers[i]);
                auto &activation = std::get<1>(_layers[i]);

                Eigen::MatrixXd dz = da.array() * activation->gradient(layer->output()).array();

                Eigen::MatrixXd prev_layer_output;
                if (i == 0)
                {
                    prev_layer_output = _input;
                }
                else
                {
                    auto &prev_layer = std::get<0>(_layers[i - 1]);
                    auto &prev_activation = std::get<1>(_layers[i - 1]);
                    prev_layer_output = prev_activation->operator()(prev_layer->output());
                }

                layer->grad_weights(dz * prev_layer_output.transpose() / _num_examples);
                layer->grad_weights(layer->grad_weights() +
                                    (_regularization_factor / _num_examples) * layer->weights());
                layer->grad_weights(dz.rowwise().mean());

                da = layer->weights().transpose() * dz;
            }
        }
    };

} // namespace NNFSCore

#endif // NEURALNETWORK_HPP
