#include "gtest/gtest.h"
#include <NNFSCore/Metrics/Metrics.hpp>
#include <Eigen/Dense>

class MetricsTest : public ::testing::Test
{
protected:
    NNFSCore::Metrics metrics_;
};

TEST_F(MetricsTest, AccuracyTest)
{
    Eigen::MatrixXd predictions(3, 3);
    predictions << 0.7, 0.2, 0.1,
        0.5, 0.1, 0.4,
        0.02, 0.9, 0.08;

    Eigen::MatrixXd labels(3, 3);
    labels << 1, 0, 0,
        0, 1, 0,
        0, 1, 0;

    double accuracy;

    metrics_.accuracy(accuracy, predictions, labels);

    ASSERT_NEAR(accuracy, 0.6666667, 1e-7);
}