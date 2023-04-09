#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <NNFSCore/Callback/Callback.hpp>
#include <NNFSCore/Callback/CSVLogger.hpp>

class CSVLoggerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        file_name_ = "test_csv_logger.csv";
        csv_logger_ = std::make_unique<NNFSCore::CSVLogger>(file_name_);
    }

    void TearDown() override
    {
        if (std::filesystem::exists(file_name_))
        {
            std::filesystem::remove(file_name_);
        }
    }

    std::string file_name_;
    std::unique_ptr<NNFSCore::CSVLogger> csv_logger_;
};

TEST_F(CSVLoggerTest, TestCSVLogger)
{
    csv_logger_->on_epoch_end(10, 0.1);

    std::ifstream file(file_name_);
    std::string line;
    std::vector<std::string> data;

    while (std::getline(file, line))
    {
        data.push_back(line);
    }

    ASSERT_EQ(data[0], "Epoch,Loss");
    ASSERT_EQ(data[1], "10,0.1");
}

TEST_F(CSVLoggerTest, FileAlreadyExists)
{

    std::ofstream file(file_name_);
    file.close();

    ASSERT_TRUE(std::filesystem::exists(file_name_));

    ASSERT_THROW(NNFSCore::CSVLogger(file_name_, false), std::runtime_error);
}

TEST_F(CSVLoggerTest, FileAlreadyExistsButOverwrite)
{
    std::ofstream file(file_name_);
    file.close();

    ASSERT_TRUE(std::filesystem::exists(file_name_));

    ASSERT_NO_THROW(NNFSCore::CSVLogger(file_name_, true));
}