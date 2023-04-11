#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>

#include <NNFSCore/Activation/ReLU.hpp>
#include <NNFSCore/Activation/Sigmoid.hpp>
#include <NNFSCore/Layer/Dense.hpp>
#include <NNFSCore/Loss/BinaryCrossEntropy.hpp>
#include <NNFSCore/Model/NeuralNetwork.hpp>
#include <NNFSCore/Callback/CSVLogger.hpp>

using namespace NNFSCore;

class NeuralNetworkTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> layers = {
            {std::make_shared<Dense>(128), std::make_shared<ReLU>()},
            {std::make_shared<Dense>(128), std::make_shared<ReLU>()},
            {std::make_shared<Dense>(10), std::make_shared<Sigmoid>()}};
        auto loss = std::make_shared<BinaryCrossEntropy>();
        double learning_rate = 0.1;

        _NeuralNetwork = std::make_unique<NeuralNetwork>(layers, loss, learning_rate);
    }

    std::unique_ptr<NeuralNetwork> _NeuralNetwork;
};

TEST_F(NeuralNetworkTest, OutputShape)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd output = _NeuralNetwork->operator()(input);
    EXPECT_EQ(output.rows(), 10);
    EXPECT_EQ(output.cols(), 16);
}

TEST_F(NeuralNetworkTest, OutputValue)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd output = _NeuralNetwork->operator()(input);
    EXPECT_NEAR(output(0, 0), 0.5, 1e-6);
}

TEST_F(NeuralNetworkTest, LearningRate)
{
    double learning_rate = 0.2;
    _NeuralNetwork->set_learning_rate(learning_rate);
    EXPECT_DOUBLE_EQ(_NeuralNetwork->get_learning_rate(), learning_rate);
}

TEST_F(NeuralNetworkTest, NumLayers)
{
    int num_layers = 3;
    EXPECT_EQ(_NeuralNetwork->get_num_layers(), num_layers);
}

TEST_F(NeuralNetworkTest, NumExamples)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(784, 16);
    _NeuralNetwork->operator()(input);
    EXPECT_EQ(_NeuralNetwork->get_num_examples(), 16);
}

TEST_F(NeuralNetworkTest, LossFunction)
{
    auto loss = _NeuralNetwork->get_loss();
    BinaryCrossEntropy *bce_loss = dynamic_cast<BinaryCrossEntropy *>(loss.get());
    ASSERT_TRUE(bce_loss != nullptr);
}

TEST_F(NeuralNetworkTest, RegularizationFactor)
{
    EXPECT_DOUBLE_EQ(_NeuralNetwork->get_regularization_factor(), 0.0);
}

TEST_F(NeuralNetworkTest, ForwardPass)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd output = _NeuralNetwork->operator()(input);
    EXPECT_EQ(output.rows(), 10);
    EXPECT_EQ(output.cols(), 16);
}

TEST_F(NeuralNetworkTest, CSVLogger)
{
    std::string log_path = "test_log.csv";
    auto csv_logger = std::make_shared<CSVLogger>(log_path, true);
    _NeuralNetwork->set_callbacks({csv_logger});
    Eigen::MatrixXd examples = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(10, 16);
    _NeuralNetwork->fit(examples, labels, 1, false);

    std::ifstream log_file(log_path);
    ASSERT_TRUE(log_file.is_open());

    std::string line;
    std::getline(log_file, line);
    EXPECT_EQ(line, "Epoch,Loss");

    std::getline(log_file, line);
    ASSERT_FALSE(line.empty());

    log_file.close();
    std::remove(log_path.c_str());
}

TEST_F(NeuralNetworkTest, Layers)
{
    Eigen::MatrixXd examples = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(10, 16);
    _NeuralNetwork->fit(examples, labels, 1, false);

    auto layers = _NeuralNetwork->get_layers();
    EXPECT_EQ(layers.size(), 3);

    Dense *dense_layer = dynamic_cast<Dense *>(std::get<0>(layers[0]).get());
    ASSERT_TRUE(dense_layer != nullptr);
    EXPECT_EQ(dense_layer->output().rows(), 128);

    ReLU *relu_activation = dynamic_cast<ReLU *>(std::get<1>(layers[0]).get());
    ASSERT_TRUE(relu_activation != nullptr);
}

TEST_F(NeuralNetworkTest, Predict)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd predictions = _NeuralNetwork->predict(input);
    EXPECT_EQ(predictions.rows(), 10);
    EXPECT_EQ(predictions.cols(), 16);
}

TEST_F(NeuralNetworkTest, Evaluate)
{
    Eigen::MatrixXd examples = Eigen::MatrixXd::Zero(784, 16);
    Eigen::MatrixXd labels = Eigen::MatrixXd::Zero(10, 16);
    double evaluation = _NeuralNetwork->evaluate(examples, labels);
    ASSERT_GT(evaluation, 0.1);
}

TEST_F(NeuralNetworkTest, Callbacks)
{
    std::vector<std::shared_ptr<Callback>> callbacks = _NeuralNetwork->get_callbacks();
    EXPECT_TRUE(callbacks.empty());
    auto csv_logger = std::make_shared<CSVLogger>("test.csv", true);
    callbacks.push_back(csv_logger);
    _NeuralNetwork->set_callbacks(callbacks);

    callbacks = _NeuralNetwork->get_callbacks();
    EXPECT_EQ(callbacks.size(), 1);
}

TEST_F(NeuralNetworkTest, LayerAccess)
{
    std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> layers = _NeuralNetwork->get_layers();
    EXPECT_EQ(layers.size(), 3);
    std::shared_ptr<Layer> layer_ptr = std::get<0>(layers[0]);
    std::shared_ptr<Activation> activation_ptr = std::get<1>(layers[0]);

    EXPECT_NE(layer_ptr, nullptr);
    EXPECT_NE(activation_ptr, nullptr);
}

TEST_F(NeuralNetworkTest, LossAccess)
{
    std::shared_ptr<Loss> loss_ptr = _NeuralNetwork->get_loss();
    EXPECT_NE(loss_ptr, nullptr);
}

TEST_F(NeuralNetworkTest, BackwardStep)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(784, 16);
    Eigen::MatrixXd labels = Eigen::MatrixXd::Random(10, 16);

    _NeuralNetwork->predict(input);

    std::vector<Eigen::MatrixXd> original_weights;
    std::vector<std::tuple<std::shared_ptr<Layer>, std::shared_ptr<Activation>>> layers = _NeuralNetwork->get_layers();
    for (const auto &layer_activation_tuple : layers)
    {
        std::shared_ptr<Layer> layer = std::get<0>(layer_activation_tuple);
        original_weights.push_back(layer->weights());
    }

    _NeuralNetwork->backward_step(labels);
    _NeuralNetwork->update();

    bool weights_updated = false;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        std::shared_ptr<Layer> layer = std::get<0>(layers[i]);
        Eigen::MatrixXd updated_weights = layer->weights();
        if (!original_weights[i].isApprox(updated_weights))
        {
            weights_updated = true;
            break;
        }
    }

    EXPECT_TRUE(weights_updated);
}

class CaptureCout : public std::streambuf
{
public:
    CaptureCout() : _oldCoutStream(std::cout.rdbuf())
    {
        std::cout.rdbuf(this);
    }

    ~CaptureCout()
    {
        std::cout.rdbuf(_oldCoutStream);
    }

    std::string getOutput() const
    {
        return _output.str();
    }

protected:
    virtual int_type overflow(int_type ch) override
    {
        if (ch != EOF)
        {
            _output.put(ch);
        }
        return ch;
    }

private:
    std::stringstream _output;
    std::streambuf *_oldCoutStream;
};

TEST_F(NeuralNetworkTest, FitVerboseOutput)
{
    // Prepare input and labels
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(784, 16);
    Eigen::MatrixXd labels = Eigen::MatrixXd::Random(10, 16);

    // Capture the output to std::cout
    CaptureCout captureCout;

    // Perform the fit function with verbose = true
    size_t epochs = 3;
    _NeuralNetwork->fit(input, labels, epochs, true);

    // Get the captured output
    std::string output = captureCout.getOutput();

    // Check if the expected output is present
    for (size_t i = 1; i <= epochs; ++i)
    {
        std::string expected_output = "Epoch: " + std::to_string(i);
        EXPECT_TRUE(output.find(expected_output) != std::string::npos) << "Expected output for epoch " << i << " not found.";
    }
}
