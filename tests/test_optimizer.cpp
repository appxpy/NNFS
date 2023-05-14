#include "gtest/gtest.h"
#include <NNFS/Core>

// Test fixture for Optimizer class
class OptimizerTest : public ::testing::Test
{
protected:
    // Set up the test fixture
    void SetUp() override
    {
        lr = 1;
        decay = 0.001;
        optimizer = std::make_unique<NNFS::SGD>(lr, decay);
    }

    // Tear down the test fixture
    void TearDown() override {}

    double lr;
    double decay;
    std::unique_ptr<NNFS::Optimizer> optimizer;
};

// Test Optimizer::pre_update_params and Optimizer::post_update_params methods
TEST_F(OptimizerTest, UpdateParams)
{
    // Check if the current learning rate is equal to the initial learning rate
    EXPECT_EQ(optimizer->current_lr(), lr);

    // Check if the current iteration count is equal to 0
    EXPECT_EQ(optimizer->iterations(), 0);

    // Update the parameters
    optimizer->pre_update_params();
    optimizer->post_update_params();

    // Check if the current learning rate is equal to the initial learning rate
    EXPECT_EQ(optimizer->current_lr(), 1);

    // Check if the current iteration count is equal to 1
    EXPECT_EQ(optimizer->iterations(), 1);

    // Update the parameters
    optimizer->pre_update_params();
    optimizer->post_update_params();

    // Check if the current learning rate is equal to the changed learning rate
    EXPECT_NEAR(optimizer->current_lr(), 0.999, 1e-5);

    // Check if the current iteration count is equal to 2
    EXPECT_EQ(optimizer->iterations(), 2);
};

// Test fixture for Adagrad class
class AdagradTest : public ::testing::Test
{
protected:
    // Set up the test fixture
    void SetUp() override
    {
        lr = 0.01;
        decay = 0.001;
        epsilon = 1e-7;
        adagrad = std::make_shared<NNFS::Adagrad>(lr, decay, epsilon);

        input_size = 2;
        output_size = 2;
        layer = std::make_shared<NNFS::Dense>(input_size, output_size);
    }

    // Tear down the test fixture
    void TearDown() override {}

    double lr;
    double decay;
    double epsilon;
    std::shared_ptr<NNFS::Adagrad> adagrad;
    std::shared_ptr<NNFS::Dense> layer;
    int input_size;
    int output_size;
};

// Test Adagrad::update_params method
TEST_F(AdagradTest, UpdateParams)
{

    // Set weights
    Eigen::MatrixXd weights{
        {.5, .5},
        {.5, .5},
    };

    layer->weights(weights);

    // Implement forward pass of the layer, setup random input matrix
    Eigen::MatrixXd inputs{
        {1, 2},
    };

    Eigen::MatrixXd before_weights = layer->weights();
    Eigen::MatrixXd before_biases = layer->biases();
    Eigen::MatrixXd before_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd before_biases_optimizer = layer->biases_optimizer();

    layer->forward(inputs, inputs);

    // Implement backward pass of the layer, setup random input matrix
    layer->backward(inputs, inputs);

    // std::cout << "before_weights: " << before_weights << std::endl;
    // std::cout << "before_biases: " << before_biases << std::endl;
    // std::cout << "before_weights_optimizer: " << before_weights_optimizer << std::endl;
    // std::cout << "before_biases_optimizer: " << before_biases_optimizer << std::endl;

    // Update parameters of the layer
    adagrad->update_params(layer);

    // Check if the parameters have been updated
    Eigen::MatrixXd after_weights = layer->weights();
    Eigen::MatrixXd after_biases = layer->biases();
    Eigen::MatrixXd after_dweights = layer->dweights();
    Eigen::MatrixXd after_dbiases = layer->dbiases();
    Eigen::MatrixXd after_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd after_biases_optimizer = layer->biases_optimizer();

    // std::cout << "after_weights: " << after_weights << std::endl;
    // std::cout << "after_biases: " << after_biases << std::endl;
    // std::cout << "after_dweights: " << after_dweights << std::endl;
    // std::cout << "after_dbiases: " << after_dbiases << std::endl;
    // std::cout << "after_weights_optimizer: " << after_weights_optimizer << std::endl;
    // std::cout << "after_biases_optimizer: " << after_biases_optimizer << std::endl;

    EXPECT_NE(before_weights, after_weights);
    EXPECT_NE(before_biases, after_biases);
    EXPECT_NE(before_weights_optimizer, after_weights_optimizer);
    EXPECT_NE(before_biases_optimizer, after_biases_optimizer);

    // Check if the parameters have been updated correctly
    Eigen::MatrixXd expected_weights{
        {0.49, 0.49},
        {0.49, 0.49},
    };

    Eigen::MatrixXd expected_biases{
        {-0.01, -0.01},
    };

    Eigen::MatrixXd expected_dweights{
        {1.5, 1.5},
        {3, 3},
    };

    Eigen::MatrixXd expected_dbiases{
        {1.5, 1.5},
    };

    Eigen::MatrixXd expected_weights_optimizer{
        {2.25, 2.25},
        {9, 9},
    };

    Eigen::MatrixXd expected_biases_optimizer{
        {2.25, 2.25},
    };

    EXPECT_TRUE(after_weights.isApprox(expected_weights, 1e-3));
    EXPECT_TRUE(after_biases.isApprox(expected_biases, 1e-3));
    EXPECT_TRUE(after_dweights.isApprox(expected_dweights, 1e-3));
    EXPECT_TRUE(after_dbiases.isApprox(expected_dbiases, 1e-3));
    EXPECT_TRUE(after_weights_optimizer.isApprox(expected_weights_optimizer, 1e-3));
    EXPECT_TRUE(after_biases_optimizer.isApprox(expected_biases_optimizer, 1e-3));
}

// Test fixture for RMSprop class
class RMSpropTest : public ::testing::Test
{
protected:
    // Set up the test fixture
    void SetUp() override
    {
        lr = 0.01;
        decay = 0.001;
        epsilon = 1e-7;
        rho = 0.9;
        rmsprop = std::make_shared<NNFS::RMSProp>(lr, decay, epsilon, rho);

        input_size = 2;
        output_size = 2;
        layer = std::make_shared<NNFS::Dense>(input_size, output_size);
    }

    // Tear down the test fixture
    void TearDown() override {}

    double lr;
    double decay;
    double epsilon;
    double rho;
    std::shared_ptr<NNFS::RMSProp> rmsprop;
    std::shared_ptr<NNFS::Dense> layer;
    int input_size;
    int output_size;
};

// Test RMSprop::update_params method
TEST_F(RMSpropTest, UpdateParams)
{

    // Set weights
    Eigen::MatrixXd weights{
        {.5, .5},
        {.5, .5},
    };

    layer->weights(weights);

    // Implement forward pass of the layer, setup random input matrix
    Eigen::MatrixXd inputs{
        {1, 2},
    };

    Eigen::MatrixXd before_weights = layer->weights();
    Eigen::MatrixXd before_biases = layer->biases();
    Eigen::MatrixXd before_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd before_biases_optimizer = layer->biases_optimizer();

    layer->forward(inputs, inputs);

    // Implement backward pass of the layer, setup random input matrix
    layer->backward(inputs, inputs);

    // std::cout << "before_weights: " << before_weights << std::endl;
    // std::cout << "before_biases: " << before_biases << std::endl;
    // std::cout << "before_weights_optimizer: " << before_weights_optimizer << std::endl;
    // std::cout << "before_biases_optimizer: " << before_biases_optimizer << std::endl;

    // Update parameters of the layer
    rmsprop->update_params(layer);

    // Check if the parameters have been updated
    Eigen::MatrixXd after_weights = layer->weights();
    Eigen::MatrixXd after_biases = layer->biases();
    Eigen::MatrixXd after_dweights = layer->dweights();
    Eigen::MatrixXd after_dbiases = layer->dbiases();
    Eigen::MatrixXd after_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd after_biases_optimizer = layer->biases_optimizer();

    // std::cout << "after_weights: " << after_weights << std::endl;
    // std::cout << "after_biases: " << after_biases << std::endl;
    // std::cout << "after_dweights: " << after_dweights << std::endl;
    // std::cout << "after_dbiases: " << after_dbiases << std::endl;
    // std::cout << "after_weights_optimizer: " << after_weights_optimizer << std::endl;
    // std::cout << "after_biases_optimizer: " << after_biases_optimizer << std::endl;

    EXPECT_NE(before_weights, after_weights);
    EXPECT_NE(before_biases, after_biases);
    EXPECT_NE(before_weights_optimizer, after_weights_optimizer);
    EXPECT_NE(before_biases_optimizer, after_biases_optimizer);

    // Check if the parameters have been updated correctly
    Eigen::MatrixXd expected_weights{
        {0.468377, 0.468377},
        {0.468377, 0.468377},
    };

    Eigen::MatrixXd expected_biases{
        {-0.0316228, -0.0316228},
    };

    Eigen::MatrixXd expected_dweights{
        {1.5, 1.5},
        {3, 3},
    };

    Eigen::MatrixXd expected_dbiases{
        {1.5, 1.5},
    };

    Eigen::MatrixXd expected_weights_optimizer{
        {0.225, 0.225},
        {0.9, 0.9},
    };

    Eigen::MatrixXd expected_biases_optimizer{
        {0.225, 0.225},
    };

    EXPECT_TRUE(after_weights.isApprox(expected_weights, 1e-3));
    EXPECT_TRUE(after_biases.isApprox(expected_biases, 1e-3));
    EXPECT_TRUE(after_dweights.isApprox(expected_dweights, 1e-3));
    EXPECT_TRUE(after_dbiases.isApprox(expected_dbiases, 1e-3));
    EXPECT_TRUE(after_weights_optimizer.isApprox(expected_weights_optimizer, 1e-3));
    EXPECT_TRUE(after_biases_optimizer.isApprox(expected_biases_optimizer, 1e-3));
}

// Test fixture for SGD class
class SGDTest : public ::testing::Test
{
protected:
    // Set up the test fixture
    void SetUp() override
    {
        lr = 0.01;
        decay = 0.001;
        momentum = 0.9;
        sgd = std::make_shared<NNFS::SGD>(lr, decay, momentum);

        input_size = 2;
        output_size = 2;
        layer = std::make_shared<NNFS::Dense>(input_size, output_size);
    }

    // Tear down the test fixture
    void TearDown() override {}

    double lr;
    double decay;
    double momentum;
    std::shared_ptr<NNFS::SGD> sgd;
    std::shared_ptr<NNFS::Dense> layer;
    int input_size;
    int output_size;
};

// Test SGD::update_params method
TEST_F(SGDTest, UpdateParams)
{

    // Set weights
    Eigen::MatrixXd weights{
        {.5, .5},
        {.5, .5},
    };

    layer->weights(weights);

    // Implement forward pass of the layer, setup random input matrix
    Eigen::MatrixXd inputs{
        {1, 2},
    };

    Eigen::MatrixXd before_weights = layer->weights();
    Eigen::MatrixXd before_biases = layer->biases();
    Eigen::MatrixXd before_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd before_biases_optimizer = layer->biases_optimizer();

    layer->forward(inputs, inputs);

    // Implement backward pass of the layer, setup random input matrix
    layer->backward(inputs, inputs);

    // std::cout << "before_weights: " << before_weights << std::endl;
    // std::cout << "before_biases: " << before_biases << std::endl;
    // std::cout << "before_weights_optimizer: " << before_weights_optimizer << std::endl;
    // std::cout << "before_biases_optimizer: " << before_biases_optimizer << std::endl;

    // Update parameters of the layer
    sgd->update_params(layer);

    // Check if the parameters have been updated
    Eigen::MatrixXd after_weights = layer->weights();
    Eigen::MatrixXd after_biases = layer->biases();
    Eigen::MatrixXd after_dweights = layer->dweights();
    Eigen::MatrixXd after_dbiases = layer->dbiases();
    Eigen::MatrixXd after_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd after_biases_optimizer = layer->biases_optimizer();

    // std::cout << "after_weights: " << after_weights << std::endl;
    // std::cout << "after_biases: " << after_biases << std::endl;
    // std::cout << "after_dweights: " << after_dweights << std::endl;
    // std::cout << "after_dbiases: " << after_dbiases << std::endl;
    // std::cout << "after_weights_optimizer: " << after_weights_optimizer << std::endl;
    // std::cout << "after_biases_optimizer: " << after_biases_optimizer << std::endl;

    EXPECT_NE(before_weights, after_weights);
    EXPECT_NE(before_biases, after_biases);
    EXPECT_NE(before_weights_optimizer, after_weights_optimizer);
    EXPECT_NE(before_biases_optimizer, after_biases_optimizer);

    // Check if the parameters have been updated correctly
    Eigen::MatrixXd expected_weights{
        {0.485, 0.485},
        {0.47, 0.47},
    };

    Eigen::MatrixXd expected_biases{
        {-0.015, -0.015},
    };

    Eigen::MatrixXd expected_dweights{
        {1.5, 1.5},
        {3, 3},
    };

    Eigen::MatrixXd expected_dbiases{
        {1.5, 1.5},
    };

    Eigen::MatrixXd expected_weights_optimizer{
        {-0.015, -0.015},
        {-0.03, -0.03},
    };

    Eigen::MatrixXd expected_biases_optimizer{
        {-0.015, -0.015},
    };

    EXPECT_TRUE(after_weights.isApprox(expected_weights, 1e-3));
    EXPECT_TRUE(after_biases.isApprox(expected_biases, 1e-3));
    EXPECT_TRUE(after_dweights.isApprox(expected_dweights, 1e-3));
    EXPECT_TRUE(after_dbiases.isApprox(expected_dbiases, 1e-3));
    EXPECT_TRUE(after_weights_optimizer.isApprox(expected_weights_optimizer, 1e-3));
    EXPECT_TRUE(after_biases_optimizer.isApprox(expected_biases_optimizer, 1e-3));
}

// Test fixture for Adam class
class AdamTest : public ::testing::Test
{
protected:
    // Set up the test fixture
    void SetUp() override
    {
        lr = 0.01;
        decay = 0.001;
        epsilon = 1e-7;
        beta1 = 0.9;
        beta2 = 0.999;
        adam = std::make_shared<NNFS::Adam>(lr, decay, epsilon, beta1, beta2);

        input_size = 2;
        output_size = 2;
        layer = std::make_shared<NNFS::Dense>(input_size, output_size);
    }

    // Tear down the test fixture
    void TearDown() override {}

    double lr;
    double decay;
    double epsilon;
    double beta1;
    double beta2;
    std::shared_ptr<NNFS::Adam> adam;
    std::shared_ptr<NNFS::Dense> layer;
    int input_size;
    int output_size;
};

// Test Adam::update_params method
TEST_F(AdamTest, UpdateParams)
{

    // Set weights
    Eigen::MatrixXd weights{
        {.5, .5},
        {.5, .5},
    };

    layer->weights(weights);

    // Implement forward pass of the layer, setup random input matrix
    Eigen::MatrixXd inputs{
        {1, 2},
    };

    Eigen::MatrixXd before_weights = layer->weights();
    Eigen::MatrixXd before_biases = layer->biases();
    Eigen::MatrixXd before_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd before_biases_optimizer = layer->biases_optimizer();
    Eigen::MatrixXd before_weights_optimizer_additional = layer->weights_optimizer_additional();
    Eigen::MatrixXd before_biases_optimizer_additional = layer->biases_optimizer_additional();

    layer->forward(inputs, inputs);

    // Implement backward pass of the layer, setup random input matrix
    layer->backward(inputs, inputs);

    // std::cout << "before_weights: " << before_weights << std::endl;
    // std::cout << "before_biases: " << before_biases << std::endl;
    // std::cout << "before_weights_optimizer: " << before_weights_optimizer << std::endl;
    // std::cout << "before_biases_optimizer: " << before_biases_optimizer << std::endl;
    // std::cout << "before_weights_optimizer_additional: " << before_weights_optimizer_additional << std::endl;
    // std::cout << "before_biases_optimizer_additional: " << before_biases_optimizer_additional << std::endl;

    // Update parameters of the layer
    adam->update_params(layer);

    // Check if the parameters have been updated
    Eigen::MatrixXd after_weights = layer->weights();
    Eigen::MatrixXd after_biases = layer->biases();
    Eigen::MatrixXd after_dweights = layer->dweights();
    Eigen::MatrixXd after_dbiases = layer->dbiases();
    Eigen::MatrixXd after_weights_optimizer = layer->weights_optimizer();
    Eigen::MatrixXd after_biases_optimizer = layer->biases_optimizer();
    Eigen::MatrixXd after_weights_optimizer_additional = layer->weights_optimizer_additional();
    Eigen::MatrixXd after_biases_optimizer_additional = layer->biases_optimizer_additional();

    // std::cout << "after_weights: " << after_weights << std::endl;
    // std::cout << "after_biases: " << after_biases << std::endl;
    // std::cout << "after_dweights: " << after_dweights << std::endl;
    // std::cout << "after_dbiases: " << after_dbiases << std::endl;
    // std::cout << "after_weights_optimizer: " << after_weights_optimizer << std::endl;
    // std::cout << "after_biases_optimizer: " << after_biases_optimizer << std::endl;
    // std::cout << "after_weights_optimizer_additional: " << after_weights_optimizer_additional << std::endl;
    // std::cout << "after_biases_optimizer_additional: " << after_biases_optimizer_additional << std::endl;

    EXPECT_NE(before_weights, after_weights);
    EXPECT_NE(before_biases, after_biases);
    EXPECT_NE(before_weights_optimizer, after_weights_optimizer);
    EXPECT_NE(before_biases_optimizer, after_biases_optimizer);
    EXPECT_NE(before_weights_optimizer_additional, after_weights_optimizer_additional);
    EXPECT_NE(before_biases_optimizer_additional, after_biases_optimizer_additional);

    // Check if the parameters have been updated correctly
    Eigen::MatrixXd expected_weights{
        {0.49, 0.49},
        {0.49, 0.49},
    };

    Eigen::MatrixXd expected_biases{
        {-0.01, -0.01},
    };

    Eigen::MatrixXd expected_dweights{
        {1.5, 1.5},
        {3, 3},
    };

    Eigen::MatrixXd expected_dbiases{
        {1.5, 1.5},
    };

    Eigen::MatrixXd expected_weights_optimizer{
        {0.00225, 0.00225},
        {0.009, 0.009},
    };

    Eigen::MatrixXd expected_biases_optimizer{
        {0.00225, 0.00225},
    };

    Eigen::MatrixXd expected_weights_optimizer_additional{
        {0.15, 0.15},
        {0.3, 0.3},
    };

    Eigen::MatrixXd expected_biases_optimizer_additional{
        {0.15, 0.15},
    };

    EXPECT_TRUE(after_weights.isApprox(expected_weights, 1e-3));
    EXPECT_TRUE(after_biases.isApprox(expected_biases, 1e-3));
    EXPECT_TRUE(after_dweights.isApprox(expected_dweights, 1e-3));
    EXPECT_TRUE(after_dbiases.isApprox(expected_dbiases, 1e-3));
    EXPECT_TRUE(after_weights_optimizer.isApprox(expected_weights_optimizer, 1e-3));
    EXPECT_TRUE(after_biases_optimizer.isApprox(expected_biases_optimizer, 1e-3));
    EXPECT_TRUE(after_weights_optimizer_additional.isApprox(expected_weights_optimizer_additional, 1e-3));
    EXPECT_TRUE(after_biases_optimizer_additional.isApprox(expected_biases_optimizer_additional, 1e-3));
}