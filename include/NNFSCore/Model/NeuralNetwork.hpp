#ifndef NEURAL_NETWORK_MODEL_HPP
#define NEURAL_NETWORK_MODEL_HPP

#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>

#include "Model.hpp"
#include "../Layer/Layer.hpp"
#include "../Layer/Dense.hpp"

#include "../Loss/Loss.hpp"
#include "../Loss/CCE_Softmax.hpp"

#include "../Metrics/Metrics.hpp"

#include "../Optimizer/Optimizer.hpp"

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
         *
         */
        NeuralNetwork(
            std::vector<std::shared_ptr<Layer>> &layers,
            std::shared_ptr<Loss> loss, std::shared_ptr<Optimizer> optimizer)
            : _layers(layers),
              _loss(loss),
              _optimizer(optimizer),
              _input(),
              _output(),
              _num_layers(layers.size()) {}

        void forward(Eigen::MatrixXd &x) override
        {

            for (int i = 0; i < _num_layers; i++)
            {
                _layers[i]->forward(x, x);
            }
        }

        void backward(Eigen::MatrixXd &predicted, const Eigen::MatrixXd &labels) override
        {
            Eigen::MatrixXd dx;
            _loss->backward(dx, predicted, labels);

            for (int i = _num_layers - 1; i >= 0; --i)
            {
                _layers[i]->backward(dx, dx);
            }

            for (int i = 0; i < _num_layers; i++)
            {
                if (_layers[i]->type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> _dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(_layers[i]);

                    _optimizer->update_params(_dense_layer);
                }
            }
        }

        void regularization_loss(double &loss)
        {
            for (int i = 0; i < _num_layers; i++)
            {
                if (_layers[i]->type == LayerType::DENSE)
                {
                    std::shared_ptr<Dense> _dense_layer = reinterpret_cast<const std::shared_ptr<Dense> &>(_layers[i]);
                    loss += _loss->regularization_loss(_dense_layer);
                }
            }
        }

        void accuracy(double &accuracy, const Eigen::MatrixXd &samples, const Eigen::MatrixXd &labels)
        {
            Eigen::MatrixXd predictions = samples;
            forward(predictions);
            Metrics::accuracy(accuracy, predictions, labels);
        }

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

        void fit(const Eigen::MatrixXd &examples,
                 const Eigen::MatrixXd &labels,
                 const Eigen::MatrixXd &test_examples,
                 const Eigen::MatrixXd &test_labels,
                 int epochs,
                 int batch_size,
                 bool verbose = true) override
        {
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

                    _loss->calculate(data_loss, batch_examples, batch_labels);

                    regularization_loss(reg_loss);

                    _optimizer->pre_update_params();
                    backward(batch_examples, batch_labels);
                    _optimizer->post_update_params();

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

                    double current_lr = _optimizer->current_lr();

                    std::cout << " - " << batch_time_total << "ms/batch"
                              << " - loss: " << std::fixed << std::setprecision(3) << total_loss
                              << " ( data: " << total_data_loss << ", reg: " << total_reg_loss
                              << " ) - train_accuracy: " << train_accuracy << " - test_accuracy: " << test_accuracy
                              << " - lr: " << current_lr
                              << std::endl;
                }
            }
        }

    private:
        std::vector<std::shared_ptr<Layer>>
            _layers;
        std::shared_ptr<Loss> _loss;
        std::shared_ptr<Optimizer> _optimizer;
        Eigen::MatrixXd _input;
        Eigen::MatrixXd _output;
        int _num_layers;
    };
} // namespace NNFSCore

#endif