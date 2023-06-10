#include "paint.h"
#include <unistd.h>

Paint::Paint(QWidget *parent) : QMainWindow(parent), ui(new Ui::paint)
{
  ui->setupUi(this);
  canvas = ui->label;
  // Set the window size to 28x28 pixels
  setFixedSize(28 * 10 + 200, 28 * 10 + 200);
  QPushButton *restartButton = ui->restartButton;
  model = NNFS::NeuralNetwork();
  char *home_dir = getenv("HOME");
  model.load(strcat(home_dir, "/EMNIST.bin"));
  canvas->installEventFilter(this);
  connect(restartButton, &QPushButton::clicked, this, &Paint::restartCanvas);
  canvas->setMouseTracking(true);
  setMouseTracking(true);
}

void Paint::restartCanvas()
{
  canvas->restartCanvas();
}

// Run predict function on mouse move if Canvas::isDrawing() is true
bool Paint::eventFilter([[maybe_unused]] QObject *obj, QEvent *event)
{
  if (event->type() == QEvent::MouseMove || event->type() == QEvent::MouseMove)
  {
    predict();
  }
  return false;
}

void Paint::predict()
{
  Eigen::MatrixXd image = canvas->getMatrix();
  std::cout << image.reshaped(28, 28).transpose() << std::endl
            << std::endl;
  Eigen::MatrixXd output = model.predict(image);
  Eigen::VectorXi labels;
  std::cout << output << std::endl
            << std::endl;
  ui->zero_bar->setValue((int)(output(0, 0) * 100));
  ui->one_bar->setValue((int)(output(0, 1) * 100));
  ui->two_bar->setValue((int)(output(0, 2) * 100));
  ui->three_bar->setValue((int)(output(0, 3) * 100));
  ui->four_bar->setValue((int)(output(0, 4) * 100));
  ui->five_bar->setValue((int)(output(0, 5) * 100));
  ui->six_bar->setValue((int)(output(0, 6) * 100));
  ui->seven_bar->setValue((int)(output(0, 7) * 100));
  ui->eight_bar->setValue((int)(output(0, 8) * 100));
  ui->nine_bar->setValue((int)(output(0, 9) * 100));
 
  ui->one->setStyleSheet("color: white; font-size: 24px;");
  ui->two->setStyleSheet("color: white; font-size: 24px;");
  ui->three->setStyleSheet("color: white; font-size: 24px;");
  ui->four->setStyleSheet("color: white; font-size: 24px;");
  ui->five->setStyleSheet("color: white; font-size: 24px;");
  ui->six->setStyleSheet("color: white; font-size: 24px;");
  ui->seven->setStyleSheet("color: white; font-size: 24px;");
  ui->eight->setStyleSheet("color: white; font-size: 24px;");
  ui->nine->setStyleSheet("color: white; font-size: 24px;");
  ui->zero->setStyleSheet("color: white; font-size: 24px;");

  NNFS::Metrics::onehotdecode(labels, output);
  std::cout << labels << std::endl;

  int largest_index = labels(0);

  auto style_sheet = QString("font-size: 32px; font-weight: bold;"
                           "color: #%1;")
  .arg(QPalette().color(QPalette::Link).rgba(), 8, 16);

  if (largest_index == 0) {
    ui->zero->setStyleSheet(style_sheet);
  } else if (largest_index == 1) {
    ui->one->setStyleSheet(style_sheet);
  } else if (largest_index == 2) {
    ui->two->setStyleSheet(style_sheet);
  } else if (largest_index == 3) {
    ui->three->setStyleSheet(style_sheet);
  } else if (largest_index == 4) {
    ui->four->setStyleSheet(style_sheet);
  } else if (largest_index == 5) {
    ui->five->setStyleSheet(style_sheet);
  } else if (largest_index == 6) {
    ui->six->setStyleSheet(style_sheet);
  } else if (largest_index == 7) {
    ui->seven->setStyleSheet(style_sheet);
  } else if (largest_index == 8) {
    ui->eight->setStyleSheet(style_sheet);
  } else if (largest_index == 9) {
    ui->nine->setStyleSheet(style_sheet);
  }
}

Paint::~Paint()
{
  delete ui;
}