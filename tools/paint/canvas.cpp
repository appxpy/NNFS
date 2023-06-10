#include "canvas.h"
#include "paint.h"

Canvas::Canvas(QWidget *parent) : QLabel(parent), drawing(false)
{
    setAttribute(Qt::WA_StaticContents);
    setFixedSize(28 * 10, 28 * 10);
    image = QImage(28, 28, QImage::Format_RGB888);
    image.fill(Qt::black);
    setPixmap(QPixmap::fromImage(image.scaled(width(), height())));
}

void Canvas::clearImage()
{
    image.fill(Qt::black);
    setPixmap(QPixmap::fromImage(image.scaled(width(), height())));
}

void Canvas::restartCanvas()
{
    clearImage();
}

Eigen::MatrixXd Canvas::getMatrix()
{
    return image_matrix;
}

void Canvas::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton || event->button() == Qt::RightButton)
    {
        lastPoint = event->pos() / 10.0;
        drawing = true;
    }
}

void Canvas::mouseMoveEvent(QMouseEvent *event)
{
    if ((event->buttons() & Qt::LeftButton || event->buttons() & Qt::RightButton) && drawing)
    {
        QPointF currentPoint = event->pos() / 10.0;
        QPainter painter(&image);
        QPen pen(drawing ? Qt::white : Qt::black, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(pen);
        painter.drawPoint(currentPoint);
        lastPoint = currentPoint;
        QImage image_scaled = image.scaled(width(), height());
        setPixmap(QPixmap::fromImage(image_scaled));
    }
}

void Canvas::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton || event->button() == Qt::RightButton)
    {
        drawing = false;
        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; col++)
            {
                image_matrix(0, row * 28 + col) = (double)qGray(image.pixel(row, col)) / 255;
            }
        }
    }
}