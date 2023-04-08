#ifndef CSV_LOGGER_HPP
#define CSV_LOGGER_HPP

#include <fstream>
#include <string>
#include <filesystem>
#include "Callback.hpp"

namespace NNFSCore
{
    /**
     * @brief Class to log epoch and loss data to a CSV file.
     */
    class CSVLogger : public Callback
    {
    public:
        /**
         * @brief Construct a new CSVLogger object.
         * @param file_path The path to the output CSV file.
         * @param overwrite Whether to overwrite the file if it already exists.
         */
        CSVLogger(const std::string &file_path, bool overwrite = false)
            : file_path_(file_path)
        {
            if (std::filesystem::exists(file_path) && !overwrite)
            {
                throw std::runtime_error("Log file already exists at " + file_path);
            }
            else
            {
                std::ofstream file(file_path_);
                file << "Epoch,Loss\n";
            }
        }

        /**
         * @brief Log the epoch and loss data to the CSV file.
         * @param epoch The epoch number.
         * @param loss The loss value at the end of the epoch.
         */
        void on_epoch_end(int epoch, float loss) override
        {
            std::ofstream file(file_path_, std::ios::app);
            file << epoch << "," << loss << "\n";
        }

    private:
        std::string file_path_;
    };

}

#endif // CSV_LOGGER_HPP
