#ifndef CANVAS_H
#define CANVAS_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <Eigen/Core>

class Canvas : public QLabel {
    Q_OBJECT

public:
    explicit Canvas(QWidget *parent = nullptr);
    void clearImage();
    void restartCanvas();
    Eigen::MatrixXd getMatrix();
protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    bool drawing;
    QImage image;
    QPointF lastPoint;
    Eigen::MatrixXd image_matrix = Eigen::MatrixXd::Zero(1, 784);
};

#endif // CANVAS_H