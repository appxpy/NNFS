# NNFS - Neural Network From Scratch in C++ 

This is a pet project developed by [appxpy](https://github.com/appxpy) and [brazenoptimist](https://github.com/brazenoptimist) for educational purposes at the Higher School of Economics (HSE) in Moscow. The project aims to build a simple neural network from scratch using C++17.

NNFS includes the header-only core library "NNFSCore" that allows users to build their custom neural networks for educational purposes. Users can configure layers and activation functions according to their needs.

## Getting Started

To get started with NNFS, clone the repository and its submodules:

```bash
git clone --recursive https://github.com/appxpy/NNFS.git
```

## Prerequisites
- C++17 compatible compiler (tested with Clang 14 and GCC 12)
- CMake (version 3.14 or higher)

## Building
To build the project, navigate to the project directory and create a build folder:

```bash
cd NNFS
mkdir build
cd build
```

And then build with cmake:

```bash
cmake ..
make
```

## Architecture
The NNFS project is organized into the following directories:

- `include/NNFSCore`: Contains the header-only core library, which includes layers, activation functions, and loss functions.
- `tools`: Contains the source code for the built executables for training and interactive paint testing.
- `tests`: Contains test cases for the core library using the Google Test framework.
- `external`: Contains submodules for the Eigen and Google Test libraries.

## Documentation
The complete documentation for the NNFS project can be found at [appxpy.github.io/NNFS](https://appxpy.github.io/NNFS).

## Dependencies

The NNFS project relies on the following libraries:

- **Eigen**: A high-level C++ library for linear algebra, which is included as a submodule in the `external/eigen` directory.
- **Google Test**: A C++ testing framework, which is included as a submodule in the `external/googletest` directory.

## Testing

The NNFS project was tested using Clang 14 and GCC 12.

To run the tests, navigate to the `build` directory and execute the following commands:

```bash
cd tests
./nnfs_tests
```

## Examples

An example of using the NNFSCore library to create a custom neural network can be found in the `tools` directory as `nnfs_example.cpp`.

## Contributing

We welcome contributions from anyone interested in improving the NNFS project. If you find a bug or have an idea for a new feature, please open an issue on the GitHub repository.

If you would like to contribute to the project, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your changes.
3. Make your changes and test them thoroughly.
4. Commit your changes with a clear and descriptive message.
5. Push your changes to your forked repository.
6. Open a pull request on the GitHub repository.

Please note that all contributions are subject to review by the project maintainers.

## License

This project is licensed under the **MIT License**. Please see the LICENSE file for more information.

---

made with ❤️ by [appxpy](https://github.com/appxpy) & [brazenoptimist](https://github.com/brazenoptimist)