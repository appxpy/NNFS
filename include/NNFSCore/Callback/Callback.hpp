#ifndef CALLBACK_HPP
#define CALLBACK_HPP

#include <Eigen/Dense>

namespace NNFSCore
{

    /**
     * @brief Abstract base class for callback functions.
     */
    class Callback
    {
    public:
        /**
         * @brief Pure virtual function to be called at the end of each epoch.
         * @param epoch The epoch number.
         * @param loss The loss value at the end of the epoch.
         */
        virtual void on_epoch_end(int epoch, float loss) = 0;
    };

}

#endif // CALLBACK_HPP
