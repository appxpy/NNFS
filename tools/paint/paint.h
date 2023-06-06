#ifndef PAINT_H
#define PAINT_H

#include "ui_paint.h"
#include "canvas.h"
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <NNFS/Core>


class Paint : public QMainWindow {
    Q_OBJECT

public:
    explicit Paint(QWidget *parent = nullptr);

    ~Paint();
private slots:
    void restartCanvas();
    void predict();

private:
    Ui::paint* ui;
    Canvas *canvas;
    NNFS::NeuralNetwork model;
};

#endif // PAINT_H
