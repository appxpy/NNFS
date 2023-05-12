#pragma once

#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <chrono>

#include "Model.hpp"
#include "../Layer/Layer.hpp"
#include "../Layer/Dense.hpp"

#include "../Activation/Activation.hpp"
#include "../Activation/ReLU.hpp"
#include "../Activation/Sigmoid.hpp"
#include "../Activation/Tanh.hpp"
#include "../Activation/Softmax.hpp"

#include "../Loss/Loss.hpp"
#include "../Loss/CCE_Softmax.hpp"

#include "../Metrics/Metrics.hpp"

#include "../Optimizer/Optimizer.hpp"

namespace NNFSCore
{

    /**
     * @class NeuralNetwork
     *
     * @brief A neural network model
     *
     * @details This class represents a neural network model capable of training on data and making predictions.
     */
    class NeuralNetwork : public Model
    {
    public:
        /**
         * @brief Constructor for NeuralNetwork
         *
         * @param[in] loss The loss function, must be a subclass of Loss, defaults to nullptr. If nullptr, training will not be possible.
         * @param[in] optimizer The optimizer, must be a subclass of Optimizer, defaults to nullptr. If nullptr, training will not be possible.
         */
        NeuralNetwork(
            std::shared_ptr<Loss> loss = nullptr, std::shared_ptr<Optimizer> optimizer = nullptr)
            : loss_object(loss),
              optimizer_object(optimizer),
              num_layers(0) {}

        /**
         * @brief Fit the neural network model to the given data
         *
         * @details This function trains the neural network model on the given data for the given number of epochs.
         *
         * @param[in] examples The training examples
         * @param[in] labels The training labels
         * @param[in] test_examples The test examples
         * @param[in] test_labels The test labels
         * @param[in] epochs The number of epochs to train for
         * @param[in] batch_size The batch size, i.e. the number of examples to train on in each batch
         * @param[in] verbose Whether to print out information about the training process
         */
        void fit(const Eigen::MatrixXd &examples,
                 const Eigen::MatrixXd &labels,
                 const Eigen::MatrixXd &test_examples,
                 const Eigen::MatrixXd &test_labels,
                 int epochs,
                 int batch_size,
                 bool verbose = true) override
        {

            if (loss_object == nullptr || optimizer_object == nullptr)
            {
                LOG_ERROR("Training is not possible for this neural network object as the loss and optimizer have not been specified.");
                return;
            }

            if (!compiled)
            {
                LOG_ERROR("Please compile the neural network object before attempting to train it.");
                return;
            }

            if (examples.cols() != input_dim || test_examples.cols() != input_dim)
            {
                LOG_ERROR("The number of columns in the examples matrix must match the input dimension of the neural network.");
                return;
            }

            if (labels.cols() != output_dim || test_labels.cols() != output_dim)
            {
                LOG_ERROR("The number of columns in the labels matrix must match the output dimension of the neural network.");
                return;
            }

            int num_examples = examples.rows();
            int num_batches = num_examples / batch_size;
            int batches_num_length = std::to_string(num_batches).length();

            for (int epoch = 1; epoch <= epochs; ++epoch)
            {
                std::cout << "Epoch " << epoch << "/" << epochs << std::endl;

                double total_loss = 0;
                double total_data_loss = 0;
                double total_reg_loss = 0;
                int batch_time_total = 0;

                auto time_start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < num_batches; ++i)
                {
                    auto batch_time_start = std::chrono::high_resolution_clock::now();
                    double batch_loss = 0;
                    double data_loss = 0;
                    double reg_loss = 0;
                    int start = i * batch_size;
                    int end = std::min(start + batch_size, num_examples);

                    Eigen::MatrixXd batch_examples = examples.middleRows(start, end - start);
                    Eigen::MatrixXd batch_labels = labels.middleRows(start, end - start);

                    forward(batch_examples);

                    loss_object->calculate(data_loss, batch_examples, batch_labels);

                    regularization_loss(reg_loss);

                    optimizer_object->pre_update_params();
                    backward(batch_examples, batch_labels);
                    optimizer_object->post_update_params();

                    if (verbose)
                    {
                        batch_loss = data_loss + reg_loss;
                        total_loss += batch_loss;

                        total_data_loss += data_loss;
                        total_reg_loss += reg_loss;

                        std::cout << " - " << std::setw(batches_num_length) << i + 1;
                        std::cout << '/' << num_batches;

                        progressbar(num_batches, i, 50);

                        auto batch_time_end = std::chrono::high_resolution_clock::now();

                        auto running = std::chrono::duration_cast<std::chrono::seconds>(batch_time_end - time_start);
                        auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_time_end - batch_time_start);

                        batch_time_total += batch_time.count();

                        std::cout << "- " << std::setw(4) << running.count();
                        std::cout << "s";

                        if (i + 1 != num_batches)
                        {
                            std::cout << "\r";
                            std::cout.flush();
                        }
                    }
                }

                if (verbose)
                {
                    double train_accuracy = 0;
                    double test_accuracy = 0;

                    accuracy(train_accuracy, examples, labels);
                    accuracy(test_accuracy, test_examples, test_labels);

                    batch_time_total /= num_batches;
                    total_loss /= num_batches;
                    total_data_loss /= num_batches;
                    total_reg_loss /= num_batches;

                    double current_lr = optimizer_object->current_lr();

                    std::cout << " - " << batch_time_total << "ms/batch"
                              << " - loss: " << std::fixed << std::setprecision(3) << total_loss
                              << " ( data: " << total_data_loss << ", reg: " << total_reg_loss
                              << " ) - train_accuracy: " << train_accuracy << " - test_accuracy: " << test_accuracy
                              << " - lr: " << current_lr
                              << std::endl;
                }
            }
        }

        /**
         * @brief Adds a layer to the neural network model
         *
         * @details This method adds a layer to the neural network model. The layer is added to the end of the neural network model.
         *
         * @param[in] layer The layer to be added to the neural network model
         */
        void add_layer(std::shared_ptr<Layer> layer)
        {
            layers.push_back(layer);
            num_layers = layers.size();
        }

        /**
         * @brief Compiles the neural network model
         *
         * @details This method compiles the neural network model by initializing the weights and biases of the layers and setting the input and output dimensions of each layer.
         */
        void compile()
        {
            compiled = false;
            input_dim = -1;
            output_dim = -1;

            if (num_layers == 0)
            {
                LOG_ERROR("Please add at least one layer to your neural network using the NNFSCore::add_layer method before compiling to ensure proper model functionality.");
                return;
            }

            LayerType prev_type;
            int prev_out = -1;

            for (int i = 0; i < num_layers; i++)
            {
                std::shared_ptr<Layer> cur_layer = layers[i];
                LayerType cur_type = cur_layer->type;

                if (cur_type == LayerType::ACTIVATION)
                {
                    if (i == 0)
                    {
                        LOG_WARNING("Applying an activation function as the first layer of a neural network can distort or lose important information in the input data. It is recommended to use a dense layer for input processing followed by activation functions in subsequent layers to preserve the integrity of the input data.");
                    }
                    else if (prev_type == cur_type)
                    {
                        LOG_WARNING("Applying multiple activation functions in a row can cause issues with learning. Consider using a different layer type or adjusting your model architecture.");
                    }
                }
                else if (cur_type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(layers[i]);

                    int cur_input;
                    int cur_output;

                    dense_layer->shape(cur_input, cur_output);

                    if (prev_out != -1 && cur_input != prev_out)
                    {
                        LOG_ERROR("Shape mismatch detected in NNFS::compile(). Previous dense layer output shape (" << prev_out << ") does not match current dense layer input shape (" << cur_input << ").");
                        return;
                    }

                    prev_out = cur_output;
                }
                else
                {
                    LOG_ERROR("Unknown layer type detected in NNFS::compile(). Please ensure that all layers in your neural network have a valid layer type and that the NNFS library supports the specified type.");
                    return;
                }

                prev_type = cur_type;
            }

            for (int i = 0; i < num_layers; i++)
            {
                std::shared_ptr<Layer> cur_layer = layers[i];
                LayerType cur_type = cur_layer->type;

                if (cur_type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(layers[i]);

                    int cur_input;
                    int cur_output;

                    dense_layer->shape(cur_input, cur_output);

                    if (input_dim == -1)
                    {
                        input_dim = cur_input;
                    }

                    output_dim = cur_output;
                }
            }

            compiled = true;
        }

        /**
         * @brief Saves the model to a file in a custom binary format. The model can be loaded using the NNFS::load method.
         *
         * @param[in] path The path to save the model to
         */
        void save(std::string path)
        {
            // Create ofstream object
            std::ofstream ofs(path, std::ios::binary);

            // Write number of layers
            ofs.write(reinterpret_cast<const char *>(&num_layers), sizeof(int));

            for (int i = 0; i < num_layers; i++)
            {
                std::shared_ptr<Layer> layer = layers[i];

                if (layer->type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(layers[i]);

                    // Schema for dense layer
                    // - int type
                    // - int n_input
                    // - int n_output
                    // - Eigen::MatrixXd weights
                    // - Eigen::MatrixXd biases
                    // - double l1_weight_regularizer
                    // - double l2_weight_regularizer
                    // - double l1_bias_regularizer
                    // - double l2_bias_regularizer
                    // - Eigen::MatrixXd weights_optimizer
                    // - Eigen::MatrixXd biases_optimizer
                    // - Eigen::MatrixXd weights_optimizer_additional
                    // - Eigen::MatrixXd biases_optimizer_additional

                    int type = static_cast<int>(layer->type);
                    int n_input;
                    int n_output;
                    dense_layer->shape(n_input, n_output);

                    Eigen::MatrixXd weights = dense_layer->weights();
                    Eigen::MatrixXd biases = dense_layer->biases();

                    double l1_weight_regularizer = dense_layer->l1_weights_regularizer();
                    double l2_weight_regularizer = dense_layer->l2_weights_regularizer();
                    double l1_bias_regularizer = dense_layer->l1_biases_regularizer();
                    double l2_bias_regularizer = dense_layer->l2_biases_regularizer();

                    Eigen::MatrixXd weights_optimizer = dense_layer->weights_optimizer();
                    Eigen::MatrixXd biases_optimizer = dense_layer->biases_optimizer();

                    Eigen::MatrixXd weights_optimizer_additional = dense_layer->weights_optimizer_additional();
                    Eigen::MatrixXd biases_optimizer_additional = dense_layer->biases_optimizer_additional();

                    // Write schema to file
                    ofs.write(reinterpret_cast<char *>(&type), sizeof(type));
                    ofs.write(reinterpret_cast<char *>(&n_input), sizeof(n_input));
                    ofs.write(reinterpret_cast<char *>(&n_output), sizeof(n_output));

                    ofs.write(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(double));
                    ofs.write(reinterpret_cast<char *>(biases.data()), biases.size() * sizeof(double));

                    ofs.write(reinterpret_cast<char *>(&l1_weight_regularizer), sizeof(double));
                    ofs.write(reinterpret_cast<char *>(&l2_weight_regularizer), sizeof(double));
                    ofs.write(reinterpret_cast<char *>(&l1_bias_regularizer), sizeof(double));
                    ofs.write(reinterpret_cast<char *>(&l2_bias_regularizer), sizeof(double));

                    ofs.write(reinterpret_cast<char *>(weights_optimizer.data()), weights_optimizer.size() * sizeof(double));
                    ofs.write(reinterpret_cast<char *>(biases_optimizer.data()), biases_optimizer.size() * sizeof(double));
                    ofs.write(reinterpret_cast<char *>(weights_optimizer_additional.data()), weights_optimizer_additional.size() * sizeof(double));
                    ofs.write(reinterpret_cast<char *>(biases_optimizer_additional.data()), biases_optimizer_additional.size() * sizeof(double));
                }
                else if (layer->type == LayerType::ACTIVATION)
                {
                    std::shared_ptr<Activation> activation_layer = reinterpret_cast<const std::shared_ptr<Activation> &>(layers[i]);

                    // Schema for activation layer
                    // - int type
                    // - int activation_type

                    int type = static_cast<int>(layer->type);
                    int activation_type = static_cast<int>(activation_layer->activation_type);

                    // Write schema to file
                    ofs.write(reinterpret_cast<char *>(&type), sizeof(type));
                    ofs.write(reinterpret_cast<char *>(&activation_type), sizeof(activation_type));
                }
                else
                {
                    LOG_ERROR("Unknown layer type detected in NNFS::save(). Please ensure that all layers in your neural network have a valid layer type and that the NNFS library supports the specified type.");
                    ofs.close();
                    std::remove(path.c_str());
                    return;
                }
            }
        }

        /**
         * @brief Loads a model from a file in a custom binary format. The model must have been saved using the NNFS::save method.
         *
         * @param[in] path The path to load the model from
         */
        void load(std::string path)
        {
            // Create ifstream object
            std::ifstream ifs(path, std::ios::binary);

            // Check if file exists
            if (!ifs.good())
            {
                LOG_ERROR("File does not exist in NNFS::load(). Please ensure that the specified file exists.");
                return;
            }

            // Clear layers
            layers.clear();

            // Read number of layers
            ifs.read(reinterpret_cast<char *>(&num_layers), sizeof(int));

            // Read layers
            for (int i = 0; i < num_layers; i++)
            {
                // Read layer type
                int type;
                ifs.read(reinterpret_cast<char *>(&type), sizeof(type));

                if (type == static_cast<int>(LayerType::DENSE))
                {
                    // Read layer shape
                    int n_input;
                    int n_output;
                    ifs.read(reinterpret_cast<char *>(&n_input), sizeof(n_input));
                    ifs.read(reinterpret_cast<char *>(&n_output), sizeof(n_output));

                    // Read weights and biases
                    Eigen::MatrixXd weights(n_input, n_output);
                    Eigen::MatrixXd biases(1, n_output);
                    ifs.read(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(double));
                    ifs.read(reinterpret_cast<char *>(biases.data()), biases.size() * sizeof(double));

                    // Read regularizers
                    double l1_weight_regularizer;
                    double l2_weight_regularizer;
                    double l1_bias_regularizer;
                    double l2_bias_regularizer;
                    ifs.read(reinterpret_cast<char *>(&l1_weight_regularizer), sizeof(double));
                    ifs.read(reinterpret_cast<char *>(&l2_weight_regularizer), sizeof(double));
                    ifs.read(reinterpret_cast<char *>(&l1_bias_regularizer), sizeof(double));
                    ifs.read(reinterpret_cast<char *>(&l2_bias_regularizer), sizeof(double));

                    // Create dense layer
                    std::shared_ptr<Dense> dense_layer = std::make_shared<Dense>(n_input, n_output, l1_weight_regularizer, l2_weight_regularizer, l1_bias_regularizer, l2_bias_regularizer);

                    // Read optimizers
                    Eigen::MatrixXd weights_optimizer(n_input, n_output);
                    Eigen::MatrixXd biases_optimizer(1, n_output);
                    Eigen::MatrixXd weights_optimizer_additional(n_input, n_output);
                    Eigen::MatrixXd biases_optimizer_additional(1, n_output);
                    ifs.read(reinterpret_cast<char *>(weights_optimizer.data()), weights_optimizer.size() * sizeof(double));
                    ifs.read(reinterpret_cast<char *>(biases_optimizer.data()), biases_optimizer.size() * sizeof(double));
                    ifs.read(reinterpret_cast<char *>(weights_optimizer_additional.data()), weights_optimizer_additional.size() * sizeof(double));
                    ifs.read(reinterpret_cast<char *>(biases_optimizer_additional.data()), biases_optimizer_additional.size() * sizeof(double));

                    // Resize all matrices
                    weights.resize(n_input, n_output);
                    biases.resize(1, n_output);
                    weights_optimizer.resize(n_input, n_output);
                    biases_optimizer.resize(1, n_output);
                    weights_optimizer_additional.resize(n_input, n_output);
                    biases_optimizer_additional.resize(1, n_output);

                    // Set weights and biases
                    dense_layer->weights(weights);
                    dense_layer->biases(biases);
                    dense_layer->weights_optimizer(weights_optimizer);
                    dense_layer->biases_optimizer(biases_optimizer);
                    dense_layer->weights_optimizer_additional(weights_optimizer_additional);
                    dense_layer->biases_optimizer_additional(biases_optimizer_additional);

                    // Add dense layer to layers
                    layers.push_back(dense_layer);
                }
                else if (type == static_cast<int>(LayerType::ACTIVATION))
                {
                    // Read activation type
                    int activation_type;
                    ifs.read(reinterpret_cast<char *>(&activation_type), sizeof(activation_type));

                    // Convert activation type to enum
                    ActivationType activation;
                    switch (activation_type)
                    {
                    case 0:
                        activation = ActivationType::RELU;
                        break;
                    case 1:
                        activation = ActivationType::SIGMOID;
                        break;
                    case 2:
                        activation = ActivationType::TANH;
                        break;
                    case 3:
                        activation = ActivationType::SOFTMAX;
                        break;
                    default:
                        activation = ActivationType::NONE;
                        break;
                    }

                    if (activation == ActivationType::NONE)
                    {
                        LOG_ERROR("Unknown activation type detected in NNFS::load(). Please ensure that all layers in your neural network have a valid activation type and that the NNFS library supports the specified type.");
                        ifs.close();
                        return;
                    }

                    // Create activation layer according to activation type, e.g std::make_shared<ReLU>() or std::make_shared<Sigmoid>() etc.;
                    std::shared_ptr<Activation> activation_layer;
                    switch (activation)
                    {
                    case ActivationType::SIGMOID:
                        activation_layer = std::make_shared<Sigmoid>();
                        break;
                    case ActivationType::TANH:
                        activation_layer = std::make_shared<Tanh>();
                        break;
                    case ActivationType::SOFTMAX:
                        activation_layer = std::make_shared<Softmax>();
                        break;
                    default:
                        activation_layer = std::make_shared<ReLU>();
                        break;
                    }

                    // Add activation layer to layers
                    layers.push_back(activation_layer);
                }
                else
                {
                    LOG_ERROR("Unknown layer type detected in NNFS::load(). Please ensure that all layers in your neural network have a valid layer type and that the NNFS library supports the specified type.");
                    ifs.close();
                    return;
                }
            }

            // If everything went well, close the file
            ifs.close();

            // Compile the neural network
            compile();
        }

        /**
         * @brief Calculates the accuracy of the neural network on the provided examples and labels.
         *
         * @param[out] accuracy Accuracy of the neural network on the provided examples and labels.
         * @param[in] examples Examples to calculate the accuracy on.
         * @param[in] labels Labels to calculate the accuracy on.
         */
        void accuracy(double &accuracy, const Eigen::MatrixXd &examples, const Eigen::MatrixXd &labels)
        {
            if (examples.cols() != input_dim || labels.cols() != output_dim)
            {
                LOG_ERROR("Input and output dimensions of the neural network do not match the dimensions of the provided samples and labels.");
                return;
            }

            Eigen::MatrixXd predictions = examples;
            forward(predictions);
            Metrics::accuracy(accuracy, predictions, labels);
        }

        /**
         * @brief Predicts the class of the provided sample(s).
         *
         * @param[in] sample Sample(s) to predict the class of.
         *
         * @return Predictions of the neural network for the provided sample(s).
         */
        Eigen::MatrixXd predict(const Eigen::MatrixXd &sample)
        {
            if (sample.cols() != input_dim)
            {
                LOG_ERROR("Input dimension of the neural network does not match the dimension of the provided sample.");
                return Eigen::MatrixXd();
            }

            Eigen::MatrixXd prediction = sample;
            forward(prediction);
            return prediction;
        }

    private:
        /**
         * @brief Implements the forward pass of the neural network.
         *
         * @param[in, out] x Input of the neural network. The output of the neural network is stored in this matrix.
         */
        void forward(Eigen::MatrixXd &x) override
        {
            for (int i = 0; i < num_layers; i++)
            {
                layers[i]->forward(x, x);
            }
        }

        /**
         * @brief Implements the backward pass of the neural network.
         *
         * @details The backward pass of the neural network is implemented by first calculating the loss of the neural network and then propagating the loss backwards through the neural network.
         *
         * @param[in] predicted Predicted output of the neural network.
         * @param[in] labels Labels of the provided examples.
         */
        void backward(Eigen::MatrixXd &predicted, const Eigen::MatrixXd &labels) override
        {
            Eigen::MatrixXd dx;
            loss_object->backward(dx, predicted, labels);

            for (int i = num_layers - 1; i >= 0; --i)
            {
                layers[i]->backward(dx, dx);
            }

            for (int i = 0; i < num_layers; i++)
            {
                if (layers[i]->type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> _dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(layers[i]);

                    optimizer_object->update_params(_dense_layer);
                }
            }
        }

        /**
         * @brief Calculates regularization loss of the neural network.
         *
         * @param[out] loss Regularization loss of the neural network.
         */
        void regularization_loss(double &loss)
        {
            for (int i = 0; i < num_layers; i++)
            {
                if (layers[i]->type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> _dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(layers[i]);
                    loss += loss_object->regularization_loss(_dense_layer);
                }
            }
        }

        /**
         * @brief Prints the progress of the training process.
         *
         * @param[in] total Total number of iterations.
         * @param[in] current Current iteration.
         * @param[in] length Length of the progress bar.
         */
        void progressbar(int total, int current, int length)
        {
            double progress = (current + 1) / (double)total;
            int pos = length * progress;

            std::cout << " [";
            for (int current = 0; current < length; ++current)
            {
                if (current < pos)
                {
                    std::cout << "=";
                }
                else if (current == pos)
                {
                    std::cout << ">";
                }
                else
                {
                    std::cout << " ";
                }
            }

            std::cout << "] " << std::setw(3) << int(progress * 100);
            std::cout << "% ";
        }

        std::vector<std::shared_ptr<Layer>> layers;
        std::shared_ptr<Loss> loss_object;
        std::shared_ptr<Optimizer> optimizer_object;
        int num_layers;
        int input_dim;
        int output_dim;
        bool compiled = false;
    };

} // namespace NNFSCore
