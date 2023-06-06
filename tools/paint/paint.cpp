#include "paint.h"
#include <unistd.h>

Paint::Paint(QWidget *parent) : QMainWindow(parent), ui(new Ui::paint) {
    ui->setupUi(this);
    canvas = ui->label;
    // Set the window size to 28x28 pixels
    setFixedSize(28 * 10 + 200, 28 * 10 + 200);
    QPushButton *restartButton = ui->restartButton;
    model = NNFS::NeuralNetwork();
    char* home_dir = getenv("HOME");
    model.load(strcat(home_dir,"/MNIST.bin"));

    connect(restartButton, &QPushButton::clicked, this, &Paint::restartCanvas);
    connect(ui->predict, &QPushButton::clicked, this, &Paint::predict);
}

void Paint::restartCanvas() {
    canvas->restartCanvas();
}

void Paint::predict() {
  Eigen::MatrixXd image = canvas->getMatrix();
  std::cout << image.reshaped(28, 28) << std::endl << std::endl;
  Eigen::MatrixXd output = model.predict(image);
  Eigen::VectorXi labels;
  std::cout << output << std::endl << std::endl;
  NNFS::Metrics::onehotdecode(labels, output);
  std::cout << labels << std::endl;
  std::cout << ">--------<" << std::endl;
}

Paint::~Paint() {
  delete ui;
}