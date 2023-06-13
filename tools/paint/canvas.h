#ifndef CANVAS_H
#define CANVAS_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <Eigen/Core>

/**
 * @brief The Canvas class represents a custom widget for drawing on a canvas.
 *
 * This class extends QLabel and provides functionality for drawing on a canvas using the mouse.
 */
class Canvas : public QLabel
{
    Q_OBJECT

public:
    /**
     * @brief Constructs a new Canvas object.
     *
     * @param parent The parent widget.
     */
    explicit Canvas(QWidget *parent = nullptr);

    /**
     * @brief Clears the image on the canvas.
     */
    void clearImage();

    /**
     * @brief Restarts the canvas by clearing the image and resetting the drawing state.
     */
    void restartCanvas();

    /**
     * @brief Gets the matrix representation of the image on the canvas.
     *
     * @return The matrix representation of the image.
     */
    Eigen::MatrixXd getMatrix();

    /**
     * @brief Checks if the user is currently drawing on the canvas.
     *
     * @return True if the user is drawing, false otherwise.
     */
    bool isDrawing() { return drawing; }

protected:
    /**
     * @brief Handles the mouse press event.
     *
     * @param event The mouse event.
     */
    void mousePressEvent(QMouseEvent *event) override;

    /**
     * @brief Handles the mouse move event.
     *
     * @param event The mouse event.
     */
    void mouseMoveEvent(QMouseEvent *event) override;

    /**
     * @brief Handles the mouse release event.
     *
     * @param event The mouse event.
     */
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    bool drawing;                                                 ///< Flag indicating whether the user is currently drawing.
    QImage image;                                                 ///< The image used as the canvas.
    QPointF lastPoint;                                            ///< The last recorded point where the mouse was pressed or moved.
    Eigen::MatrixXd image_matrix = Eigen::MatrixXd::Zero(1, 784); ///< The matrix representation of the image on the canvas.
};

#endif // CANVAS_H
