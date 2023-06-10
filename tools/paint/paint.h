#ifndef PAINT_H
#define PAINT_H

#include "ui_paint.h"
#include "canvas.h"
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <NNFS/Core>

/**
 * @brief The Paint class represents a painting application.
 *
 * @details The Paint class is a QMainWindow that provides a user interface for a painting application.
 * It allows the user to draw on a canvas and provides functionality to restart the canvas and predict using a neural network model.
 */
class Paint : public QMainWindow
{
    Q_OBJECT

public:
    /**
     * @brief Constructs a new Paint object.
     *
     * @param parent The parent QWidget.
     */
    explicit Paint(QWidget *parent = nullptr);

    /**
     * @brief Destroys the Paint object.
     */
    ~Paint();

protected:
    /**
     * @brief Filters events for the Paint object.
     *
     * @param obj The object that sent the event.
     * @param event The event that occurred.
     * @return true if the event was handled, false otherwise.
     */
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    /**
     * @brief Restarts the canvas.
     */
    void restartCanvas();

    /**
     * @brief Performs a prediction using the neural network model.
     */
    void predict();

private:
    Ui::paint *ui;               // User interface
    Canvas *canvas;              // Painting canvas
    NNFS::NeuralNetwork model;   // Neural network model
};

#endif // PAINT_H
